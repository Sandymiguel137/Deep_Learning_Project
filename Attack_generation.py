#!/usr/bin/env python3
"""
Generate the SMART-DS/OpenDSS attack dataset with boundary-stress samples.

1. PV voltages are softly regulated near the lower volt-var deadband edge.
2. Slow per-load net-load sinusoids are superimposed so normal samples contain
   legitimate VV/VW control transients near the deadband boundary.

Default outputs are written next to this script:
  - Y_norm_sparse.npy
  - PhaseMatrix.npy
  - bus_phase_mapping.json
  - Vall_ReIm_boundary_stress.npy
  - attack_label_boundary_stress.npy
  - VphasorFDI_boundary_stress.npy
  - AttackLabelFDI_boundary_stress.npy
  - metadata_boundary_stress.npy
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd
import scipy.linalg as spla
import scipy.sparse as sps
from scipy.sparse import csc_matrix, diags
from scipy.sparse.linalg import svds

try:
    import opendssdirect as dss
except ImportError as exc:
    raise SystemExit(
        "opendssdirect is required to generate the OpenDSS dataset. "
        "Install it in the project environment before running this script."
    ) from exc


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from data_loader import data_loader_FDIPhy  # noqa: E402


PV_SENSOR_SEEDS = [
    "p4ulv5.1", "p4ulv5.2", "p4ulv6.1", "p4ulv6.2", "p4ulv10.1", "p4ulv10.2",
    "p4ulv17.1", "p4ulv17.2", "p4ulv18.1", "p4ulv18.2", "p4ulv20",
    "p4ulv21.1", "p4ulv21.2", "p4ulv22.1", "p4ulv22.2", "p4ulv23.1",
    "p4ulv23.2", "p4ulv24.1", "p4ulv24.2", "p4ulv25.1", "p4ulv25.2",
    "p4ulv26.1", "p4ulv26.2", "p4ulv27", "p4ulv28", "p4ulv30.1",
    "p4ulv30.2", "p4ulv33", "p4ulv34.1", "p4ulv34.2",
]


@dataclass
class GeneratorConfig:
    feeder: str = "p4uhs0_4--p4udt4"
    substation: str = "p4uhs0_4"
    sampling_rate: int = 20
    max_timepoints: int | None = None
    seed: int = 100
    num_pv: int = 30
    sv_power: float = 2.96
    sensor_count: int = 120
    svd_rank: int = 80
    fdi_enabled: bool = True
    boundary_stress: bool = True
    lower_deadband_edge: float = 0.97
    stress_amplitude_min: float = 0.015
    stress_amplitude_max: float = 0.055
    stress_period_min: float = 8.0
    stress_period_max: float = 28.0
    voltage_feedback_gain: float = 2.5
    voltage_feedback_clip: float = 0.18
    compromised_curve: Tuple[float, float, float, float, float] = (0.98, 0.99, 1.01, 1.02, 1.10)


class PersistentPhysicalAttack:
    def __init__(
        self,
        pv_phase2_flags: List[bool],
        rng: np.random.Generator,
        attack_fraction_range: Tuple[float, float] = (0.3, 0.4),
        attack_duration_range: Tuple[int, int] = (10, 30),
        cooldown_duration_range: Tuple[int, int] = (50, 100),
        compromised_curve: Iterable[float] = (0.98, 0.99, 1.01, 1.02, 1.10),
    ) -> None:
        self.pv_phase2_flags = pv_phase2_flags
        self.rng = rng
        self.attack_fraction_range = attack_fraction_range
        self.attack_duration_range = attack_duration_range
        self.cooldown_duration_range = cooldown_duration_range
        self.compromised_curve = np.asarray(tuple(compromised_curve), dtype=float)
        self.phase2_indices = [idx for idx, is_p2 in enumerate(pv_phase2_flags) if is_p2]
        self.state = {
            idx: {"active": False, "remaining": 0, "cooldown": 0}
            for idx in self.phase2_indices
        }

    def step(self, eta: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        eta_comm = eta.copy()
        attack_label = np.zeros(eta.shape[0], dtype=int)
        if not self.phase2_indices:
            return eta_comm, attack_label

        attack_fraction = self.rng.uniform(*self.attack_fraction_range)
        num_to_attack = max(1, int(np.floor(len(self.phase2_indices) * attack_fraction)))
        eligible = [
            idx for idx in self.phase2_indices
            if not self.state[idx]["active"] and self.state[idx]["cooldown"] <= 0
        ]
        new_attacks = set(
            self.rng.choice(eligible, size=min(num_to_attack, len(eligible)), replace=False).tolist()
            if eligible else []
        )

        for idx in self.phase2_indices:
            state = self.state[idx]
            if state["active"]:
                state["remaining"] -= 1
                if state["remaining"] <= 0:
                    state["active"] = False
                    state["cooldown"] = int(self.rng.integers(*self.cooldown_duration_range))
            else:
                if state["cooldown"] > 0:
                    state["cooldown"] -= 1
                if idx in new_attacks:
                    state["active"] = True
                    state["remaining"] = int(self.rng.integers(*self.attack_duration_range))

            if state["active"]:
                eta_comm[idx, 1, :] = self.compromised_curve
                attack_label[idx] = 1

        return eta_comm, attack_label


def master_dss_path(config: GeneratorConfig) -> Path:
    return (
        PROJECT_ROOT / "P4U" / "scenarios" / "base_timeseries" / "opendss_no_loadshapes"
        / config.substation / config.feeder / "Master.dss"
    )


def loads_dss_path(config: GeneratorConfig) -> Path:
    return master_dss_path(config).with_name("Loads.dss")


def load_data_dir() -> Path:
    return PROJECT_ROOT / "P4U" / "load_data"


def redirect_master(config: GeneratorConfig) -> None:
    dss.Basic.ClearAll()
    result = dss.run_command("Redirect " + str(master_dss_path(config)))
    if result:
        print(result)


def normalize_ybus() -> Tuple[csc_matrix, List[str], np.ndarray]:
    y_node_order = list(dss.Circuit.YNodeOrder())
    y_sparse = sps.csc_matrix(dss.YMatrix.getYsparse())
    dmat = diags(y_sparse.diagonal(), format="csc")
    inv_sqrt_diag = 1 / np.sqrt(dmat.diagonal())
    inv_sqrt_diag[~np.isfinite(inv_sqrt_diag)] = 0
    dmat_inv_sqrt = diags(inv_sqrt_diag, format="csc")
    y_norm = (dmat_inv_sqrt @ y_sparse @ dmat_inv_sqrt).toarray()
    y_norm.real[np.abs(y_norm.real) < 1e-8] = 0
    y_norm.imag[np.abs(y_norm.imag) < 1e-8] = 0
    y_norm_sparse = csc_matrix(y_norm)
    u, _, _ = spla.svd(y_norm_sparse.toarray(), full_matrices=False)
    return y_norm_sparse, y_node_order, u


def phase_matrix(y_node_order: List[str]) -> np.ndarray:
    phases = np.zeros((len(y_node_order), 3), dtype=int)
    for idx, node_name in enumerate(y_node_order):
        if ".1" in node_name:
            phases[idx, 0] = 1
        if ".2" in node_name:
            phases[idx, 1] = 1
        if ".3" in node_name:
            phases[idx, 2] = 1
    return phases


def bus_phase_mapping() -> Dict[str, List[int]]:
    mapping = {}
    for bus_name in dss.Circuit.AllBusNames():
        dss.Circuit.SetActiveBus(bus_name)
        mapping[bus_name] = list(dss.Bus.Nodes())
    return mapping


def parse_load_profiles(config: GeneratorConfig) -> Dict[str, Tuple[str, float]]:
    profile_map: Dict[str, Tuple[str, float]] = {}
    with loads_dss_path(config).open() as f_load:
        for row in f_load:
            name = None
            profile = None
            multiplier = 1.0
            for token in row.split():
                if token.startswith("Load."):
                    name = token.split(".", 1)[1]
                    if name.endswith("_1") or name.endswith("_2"):
                        multiplier = 0.5
                if token.startswith("!yearly"):
                    profile_raw = token.split("=", 1)[1]
                    if "mesh" in profile_raw:
                        profile = "mesh"
                    else:
                        parts = profile_raw.split("_")
                        profile = parts[0] + "_" + parts[2] + ".parquet"
                if name is not None and profile is not None:
                    profile_map[name] = (profile, multiplier)
    return profile_map


def load_profile_arrays(
    profile_map: Dict[str, Tuple[str, float]],
    max_timepoints: int | None,
) -> Tuple[Dict[str, Tuple[np.ndarray, np.ndarray]], int]:
    cache: Dict[str, pd.DataFrame | None] = {}
    arrays: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}
    total_timepoints = None

    for load_name, (profile, multiplier) in profile_map.items():
        if profile == "mesh":
            if total_timepoints is None:
                raise ValueError("A mesh profile appeared before a real profile established length.")
            arrays[load_name] = (np.zeros(total_timepoints), np.zeros(total_timepoints))
            continue

        if profile not in cache:
            df = pd.read_parquet(load_data_dir() / profile)
            if max_timepoints is not None:
                df = df.iloc[:max_timepoints]
            cache[profile] = df
            if total_timepoints is None:
                total_timepoints = len(df)
        df = cache[profile]
        assert df is not None
        arrays[load_name] = (
            df["total_site_electricity_kw"].to_numpy(dtype=float) * multiplier,
            df["total_site_electricity_kvar"].to_numpy(dtype=float) * multiplier,
        )

    if total_timepoints is None:
        raise ValueError("No usable load profiles were found.")
    return arrays, total_timepoints


def load_values_at_time(
    profile_arrays: Dict[str, Tuple[np.ndarray, np.ndarray]],
    time_idx: int,
) -> Tuple[Dict[str, float], Dict[str, float]]:
    active = {name: float(values[0][time_idx]) for name, values in profile_arrays.items()}
    reactive = {name: float(values[1][time_idx]) for name, values in profile_arrays.items()}
    return active, reactive


def make_stress_parameters(
    load_names: List[str],
    total_timepoints: int,
    config: GeneratorConfig,
    rng: np.random.Generator,
) -> Dict[str, Tuple[float, float, float]]:
    periods = rng.uniform(config.stress_period_min, config.stress_period_max, size=len(load_names))
    amplitudes = rng.uniform(config.stress_amplitude_min, config.stress_amplitude_max, size=len(load_names))
    phases = rng.uniform(0, 2 * np.pi, size=len(load_names))
    periods = np.minimum(periods, max(config.sampling_rate, total_timepoints / 2))
    return {
        name: (float(amplitudes[i]), float(2 * np.pi / periods[i]), float(phases[i]))
        for i, name in enumerate(load_names)
    }


def apply_boundary_stress(
    active: Dict[str, float],
    reactive: Dict[str, float],
    stress_params: Dict[str, Tuple[float, float, float]],
    time_idx: int,
    mean_pv_voltage: float,
    config: GeneratorConfig,
) -> Tuple[Dict[str, float], Dict[str, float]]:
    if not config.boundary_stress:
        return dict(active), dict(reactive)

    feedback = 1.0 + config.voltage_feedback_gain * (mean_pv_voltage - config.lower_deadband_edge)
    feedback = float(np.clip(
        feedback,
        1.0 - config.voltage_feedback_clip,
        1.0 + config.voltage_feedback_clip,
    ))

    active_stressed: Dict[str, float] = {}
    reactive_stressed: Dict[str, float] = {}
    for name, kw in active.items():
        amp, omega, phase = stress_params[name]
        sinusoid = 1.0 + amp * np.sin(omega * time_idx + phase)
        factor = max(0.05, feedback * sinusoid)
        active_stressed[name] = kw * factor
        reactive_stressed[name] = reactive[name] * factor
    return active_stressed, reactive_stressed


def is_phase2(feeder_name: str) -> bool:
    try:
        dss.Loads.Name(feeder_name)
        return 2 in dss.CktElement.NodeOrder()
    except Exception as exc:
        print(f"Warning: could not resolve feeder {feeder_name}: {exc}")
        return False


def get_all_node_voltage_re_im_map() -> Dict[str, Tuple[float, float]]:
    voltage_map = {}
    for bus_name in dss.Circuit.AllBusNames():
        dss.Circuit.SetActiveBus(bus_name)
        pu_values = dss.Bus.PuVoltage()
        phases = dss.Bus.Nodes()
        for i, phase in enumerate(phases):
            voltage_map[f"{bus_name.lower()}.{phase}"] = (pu_values[2 * i], pu_values[2 * i + 1])
    return voltage_map


def get_pv_voltage_mag(pv_feeder_reshape: List[str]) -> np.ndarray:
    voltage_map = get_all_node_voltage_re_im_map()
    magnitudes = np.zeros(len(pv_feeder_reshape), dtype=float)
    for idx, feeder in enumerate(pv_feeder_reshape):
        key = feeder.lower()
        if key in voltage_map:
            real, imag = voltage_map[key]
            magnitudes[idx] = abs(complex(real, imag))
    return magnitudes


def vv_func(vm_pv_lp_t1: np.ndarray, qv_power: np.ndarray, eta_phase: np.ndarray) -> np.ndarray:
    eta_1 = eta_phase[:, 0]
    eta_2 = eta_phase[:, 1]
    eta_3 = eta_phase[:, 2]
    eta_4 = eta_phase[:, 3]
    qv_inj = np.zeros_like(vm_pv_lp_t1)

    for i, voltage in enumerate(vm_pv_lp_t1):
        qmax = qv_power[i]
        if voltage <= eta_1[i]:
            qv_inj[i] = qmax
        elif voltage <= eta_2[i]:
            qv_inj[i] = (eta_2[i] - voltage) / (eta_2[i] - eta_1[i]) * qmax
        elif voltage <= eta_3[i]:
            qv_inj[i] = 0
        elif voltage <= eta_4[i]:
            qv_inj[i] = -(voltage - eta_3[i]) / (eta_4[i] - eta_3[i]) * qmax
        else:
            qv_inj[i] = -qmax
    return qv_inj


def vv_vw_func(
    vm_pv: np.ndarray,
    eta: np.ndarray,
    pv_power: np.ndarray,
    sv_power: float,
    vm_pv_lp_t0: np.ndarray,
    pinj_pv_t0: np.ndarray,
    qinj_pv_t0: np.ndarray,
    target_phase: int = 2,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    tau_c = 0.98
    tau_o = 0.98
    phase_idx = target_phase - 1
    vm_pv_lp_t1 = vm_pv_lp_t0 + tau_c * (vm_pv - vm_pv_lp_t0)

    eta_4 = eta[:, phase_idx, 3]
    eta_5 = eta[:, phase_idx, 4]
    pw_inj = np.zeros_like(vm_pv_lp_t1)
    for i, voltage in enumerate(vm_pv_lp_t1):
        if voltage <= eta_4[i]:
            pw_inj[i] = pv_power[i]
        elif voltage <= eta_5[i]:
            pw_inj[i] = (eta_5[i] - voltage) / (eta_5[i] - eta_4[i]) * pv_power[i]
        else:
            pw_inj[i] = 0

    qv_power = np.sqrt(np.maximum(sv_power**2 - pw_inj**2, 0))
    qv_inj = vv_func(vm_pv_lp_t1, qv_power, eta[:, phase_idx, :])

    pinj_pv_t1 = pinj_pv_t0 + tau_o * (pw_inj - pinj_pv_t0)
    qinj_pv_t1 = qinj_pv_t0 + tau_o * (qv_inj - qinj_pv_t0)
    return pinj_pv_t1 + 1j * qinj_pv_t1, vm_pv_lp_t1, pinj_pv_t1, qinj_pv_t1


def modify_mpc(
    pv_inj: np.ndarray,
    pv_feeder: List[str],
    active_power: Dict[str, float],
    reactive_power: Dict[str, float],
) -> None:
    injection_map = {
        feeder: (float(np.real(pv_inj[i])), float(np.imag(pv_inj[i])))
        for i, feeder in enumerate(pv_feeder)
    }
    dss.Loads.First()
    while True:
        name = dss.Loads.Name()
        base_kw = active_power.get(name, 0.0)
        base_kvar = reactive_power.get(name, 0.0)
        inj_kw, inj_kvar = injection_map.get(name, (0.0, 0.0))
        dss.Loads.kW(base_kw + inj_kw)
        dss.Loads.kvar(base_kvar + inj_kvar)
        if not dss.Loads.Next() > 0:
            break


def optimal_placement_greedy(u_k: np.ndarray, sensor_count: int, seed_indices: List[int]) -> List[int]:
    selected = list(dict.fromkeys(seed_indices))
    remaining = [idx for idx in range(u_k.shape[0]) if idx not in selected]
    while len(selected) < sensor_count and remaining:
        scores = []
        for idx in remaining:
            matrix = u_k[selected + [idx], :]
            try:
                score = svds(matrix, k=1, which="SM", return_singular_vectors=False)[0]
            except Exception:
                singular_values = np.linalg.svd(matrix, compute_uv=False)
                score = np.min(singular_values) if singular_values.size else 0
            scores.append(score)
        best_pos = int(np.argmax(scores))
        selected.append(remaining.pop(best_pos))
        if len(selected) % 10 == 0:
            print(f"Selected {len(selected)}/{sensor_count} sensors")
    return selected


def select_sensor_indices(y_node_order: List[str], u: np.ndarray, config: GeneratorConfig) -> Tuple[List[str], List[int]]:
    y_lower = [node.lower() for node in y_node_order]
    fixed_sensors = [
        y_node_order[y_lower.index(sensor.lower())]
        for sensor in PV_SENSOR_SEEDS
        if sensor.lower() in y_lower
    ]
    seed_indices = [y_node_order.index(name) for name in fixed_sensors]
    u_k = u[:, : min(config.svd_rank, u.shape[1])]
    selected_indices = optimal_placement_greedy(u_k, config.sensor_count, seed_indices)
    selected_names = [y_node_order[idx] for idx in selected_indices]
    return selected_names, selected_indices


def generate_raw_dataset(config: GeneratorConfig, output_dir: Path) -> Dict[str, object]:
    rng = np.random.default_rng(config.seed)
    redirect_master(config)
    print(f"OpenDSS circuit: {dss.Circuit.NumBuses()} buses, {dss.Circuit.NumNodes()} nodes")

    y_norm_sparse, y_node_order, u = normalize_ybus()
    np.save(output_dir / "Y_norm_sparse.npy", y_norm_sparse)
    np.save(output_dir / "PhaseMatrix.npy", phase_matrix(y_node_order))
    with (output_dir / "bus_phase_mapping.json").open("w") as f:
        json.dump(bus_phase_mapping(), f, indent=2)

    selected_sensor_names, sensor_location_indices = select_sensor_indices(y_node_order, u, config)
    print(f"mu-PMU sensor count: {len(selected_sensor_names)}")

    profile_map = parse_load_profiles(config)
    profile_arrays, total_timepoints = load_profile_arrays(profile_map, config.max_timepoints)
    stress_params = make_stress_parameters(list(profile_arrays), total_timepoints, config, rng)

    pv_feeder = list(profile_map.keys())[: config.num_pv]
    num_pv = len(pv_feeder)
    pv_phase2_flags = [is_phase2(feeder) for feeder in pv_feeder]
    pv_feeder_reshape = [feeder.replace("load_", "").replace("_", ".").lower() for feeder in pv_feeder]
    eta_base = np.array([0.95, 0.97, 1.03, 1.05, 1.10])
    eta = np.tile(eta_base, (num_pv, 3, 1))
    pv_power = rng.uniform(1.0, 1.2, size=(total_timepoints, num_pv))
    attack = PersistentPhysicalAttack(
        pv_phase2_flags,
        rng,
        compromised_curve=config.compromised_curve,
    )

    dss.run_command("Solve")
    ori_vm_mag = get_pv_voltage_mag(pv_feeder_reshape)
    node_keys = [node.lower() for node in y_node_order]
    num_node = len(node_keys)

    voltage_store = np.empty((total_timepoints, num_node, config.sampling_rate), dtype=complex)
    attack_label_store = np.zeros((total_timepoints, num_pv), dtype=int)
    crossing_count = 0

    for time_idx in range(total_timepoints):
        active_power, reactive_power = load_values_at_time(profile_arrays, time_idx)
        mean_reference_voltage = float(np.mean(ori_vm_mag[ori_vm_mag > 0])) if np.any(ori_vm_mag > 0) else 1.0
        active_power, reactive_power = apply_boundary_stress(
            active_power,
            reactive_power,
            stress_params,
            time_idx,
            mean_reference_voltage,
            config,
        )

        eta_comm, attack_labels_t = attack.step(eta)
        attack_label_store[time_idx, :] = attack_labels_t
        attacked_pvs = [pv_feeder_reshape[i] for i in range(num_pv) if attack_labels_t[i] == 1]
        print(f"time index {time_idx + 1}/{total_timepoints}; attacked PVs: {attacked_pvs}")

        pinj_pv_t0 = np.zeros(num_pv)
        qinj_pv_t0 = np.zeros(num_pv)
        vm_pv_lp_t0 = ori_vm_mag.copy()
        vm_pv_mag = ori_vm_mag.copy()
        previous_boundary_side = vm_pv_mag < config.lower_deadband_edge

        for sample in range(config.sampling_rate):
            pv_inj, vm_pv_lp_t1, pinj_pv_t1, qinj_pv_t1 = vv_vw_func(
                vm_pv_mag,
                eta_comm,
                pv_power[time_idx, :],
                config.sv_power,
                vm_pv_lp_t0,
                pinj_pv_t0,
                qinj_pv_t0,
            )
            vm_pv_lp_t0 = vm_pv_lp_t1
            pinj_pv_t0 = pinj_pv_t1
            qinj_pv_t0 = qinj_pv_t1

            modify_mpc(pv_inj, pv_feeder, active_power, reactive_power)
            dss.run_command("Solve")
            voltage_map = get_all_node_voltage_re_im_map()
            for node_idx, node in enumerate(node_keys):
                real, imag = voltage_map[node]
                voltage_store[time_idx, node_idx, sample] = complex(real, imag)

            vm_pv_mag = get_pv_voltage_mag(pv_feeder_reshape)
            current_boundary_side = vm_pv_mag < config.lower_deadband_edge
            crossing_count += int(np.count_nonzero(current_boundary_side != previous_boundary_side))
            previous_boundary_side = current_boundary_side

    raw_voltage_path = output_dir / "Vall_ReIm_boundary_stress.npy"
    raw_label_path = output_dir / "attack_label_boundary_stress.npy"
    np.save(raw_voltage_path, voltage_store)
    np.save(raw_label_path, attack_label_store)

    return {
        "raw_voltage_path": str(raw_voltage_path),
        "raw_label_path": str(raw_label_path),
        "y_norm_sparse": y_norm_sparse,
        "y_node_order": y_node_order,
        "pv_feeders": pv_feeder,
        "sensor_nodes": selected_sensor_names,
        "sensor_location_indices": sensor_location_indices,
        "total_timepoints": total_timepoints,
        "num_boundary_crossings": crossing_count,
    }


def postprocess_fdi(
    raw_voltage_path: Path,
    raw_label_path: Path,
    y_norm_sparse: csc_matrix,
    sensor_location_indices: List[int],
    y_node_order: List[str],
    output_dir: Path,
    fdi_enabled: bool,
) -> Tuple[Path, Path]:
    data = data_loader_FDIPhy(str(raw_voltage_path), str(raw_label_path))
    if fdi_enabled:
        data.state_est_withFDI(y_norm_sparse, sensor_location_indices)
    else:
        data.state_estimation(y_norm_sparse, sensor_location_indices)

    _, dim_bus, _ = data.data_recover.shape
    _, dim_label = data.label_truth.shape
    assert dim_bus == len(y_node_order), "Mismatch between recovered buses and Y_NodeOrder"
    expected_labels = len(sensor_location_indices) + 30 if fdi_enabled else 30
    assert dim_label == expected_labels, f"Expected {expected_labels} labels, got {dim_label}"

    voltage_path = output_dir / "VphasorFDI_boundary_stress.npy"
    label_path = output_dir / "AttackLabelFDI_boundary_stress.npy"
    np.save(voltage_path, data.data_recover)
    np.save(label_path, data.label_truth)
    return voltage_path, label_path


def save_metadata(metadata: Dict[str, object], config: GeneratorConfig, output_dir: Path) -> None:
    metadata = dict(metadata)
    metadata["config"] = asdict(config)
    metadata["boundary_stress_description"] = {
        "target": "lower volt-var deadband edge",
        "lower_deadband_edge": config.lower_deadband_edge,
        "net_load_perturbation": "A_n * sin(omega_n * t + phi_n) per load",
        "voltage_feedback": "load scaling nudges mean PV voltage toward lower_deadband_edge",
    }
    np.save(output_dir / "metadata_boundary_stress.npy", metadata)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=SCRIPT_DIR)
    parser.add_argument("--max-timepoints", type=int, default=None, help="Limit runtime for quick tests.")
    parser.add_argument("--sampling-rate", type=int, default=20)
    parser.add_argument("--seed", type=int, default=100)
    parser.add_argument("--no-fdi", action="store_true", help="Run state estimation without FDI injection.")
    parser.add_argument("--no-boundary-stress", action="store_true", help="Match the original notebook load stream.")
    parser.add_argument("--skip-raw", action="store_true", help="Reuse existing raw boundary-stress .npy files.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    config = GeneratorConfig(
        sampling_rate=args.sampling_rate,
        max_timepoints=args.max_timepoints,
        seed=args.seed,
        fdi_enabled=not args.no_fdi,
        boundary_stress=not args.no_boundary_stress,
    )

    if args.skip_raw:
        raw_voltage_path = output_dir / "Vall_ReIm_boundary_stress.npy"
        raw_label_path = output_dir / "attack_label_boundary_stress.npy"
        if not raw_voltage_path.exists() or not raw_label_path.exists():
            raise FileNotFoundError("Raw files are missing; rerun without --skip-raw.")
        redirect_master(config)
        y_norm_sparse, y_node_order, u = normalize_ybus()
        sensor_nodes, sensor_location_indices = select_sensor_indices(y_node_order, u, config)
        metadata = {
            "raw_voltage_path": str(raw_voltage_path),
            "raw_label_path": str(raw_label_path),
            "y_norm_sparse": y_norm_sparse,
            "y_node_order": y_node_order,
            "pv_feeders": list(parse_load_profiles(config).keys())[: config.num_pv],
            "sensor_nodes": sensor_nodes,
            "sensor_location_indices": sensor_location_indices,
            "total_timepoints": int(np.load(raw_voltage_path, mmap_mode="r").shape[0]),
            "num_boundary_crossings": None,
        }
    else:
        metadata = generate_raw_dataset(config, output_dir)
        raw_voltage_path = Path(str(metadata["raw_voltage_path"]))
        raw_label_path = Path(str(metadata["raw_label_path"]))

    voltage_path, label_path = postprocess_fdi(
        raw_voltage_path,
        raw_label_path,
        metadata["y_norm_sparse"],
        metadata["sensor_location_indices"],
        metadata["y_node_order"],
        output_dir,
        config.fdi_enabled,
    )
    metadata["processed_voltage_path"] = str(voltage_path)
    metadata["processed_label_path"] = str(label_path)
    save_metadata(metadata, config, output_dir)

    print("Generation complete.")
    print(f"Raw voltage: {raw_voltage_path}")
    print(f"Raw labels:  {raw_label_path}")
    print(f"FDI voltage: {voltage_path}")
    print(f"FDI labels:  {label_path}")


if __name__ == "__main__":
    main()
