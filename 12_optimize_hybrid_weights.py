"""Optimize hybrid route-generation weights on validation routes.

This final experiment script loads a trained next-stop preference model, streams
Amazon travel-time matrices route by route, generates greedy routes, searches
hybrid cost weights, and writes CSV/JSON outputs for dissertation reporting.
"""
from __future__ import annotations
import argparse
import csv
import importlib.util
import json
import math
import random
import shutil
import sys
import warnings
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Sequence
import joblib
import numpy as np
import pandas as pd
REPO_ROOT = Path(__file__).resolve().parent
SRC_DIR = REPO_ROOT / 'src'
if SRC_DIR.exists():
    sys.path.insert(0, str(SRC_DIR))
from last_mile_cleaning.clean_pipeline import stream_top_level_object

def load_repo_script(path, module_name):
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f'Could not load helper script: {path}')
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module
PAIRWISE = load_repo_script(REPO_ROOT / '09_create_full_pairwise_samples.py', 'pairwise_helpers')
VALID_SPLITS = ('train', 'validation', 'test')
SOURCE_PRIORITY = ('training_build', 'training_apply', 'evaluation_apply')
UNKNOWN_ZONE = 'UNKNOWN_ZONE'
DEFAULT_CONFIG = Path('config/config_final.yaml')
DEFAULT_BASE = Path('/content/drive/MyDrive/dissertation/amazon_last_mile')
DEFAULT_PROCESSED = DEFAULT_BASE / 'processed_outputs'
DEFAULT_FINAL = DEFAULT_PROCESSED / 'final_cleaned'
DEFAULT_MODEL_DIR = DEFAULT_BASE / 'final_experiment_outputs/model_outputs_full'
DEFAULT_OUTPUT_DIR = DEFAULT_BASE / 'final_experiment_outputs/hybrid_weight_search'
DEFAULT_MATRICES = {'training_build': DEFAULT_BASE / 'almrrc2021-data-training/model_build_inputs/travel_times.json', 'training_apply': DEFAULT_BASE / 'almrrc2021-data-training/model_apply_inputs/new_travel_times.json', 'evaluation_apply': DEFAULT_BASE / 'almrrc2021-data-evaluation/model_apply_inputs/eval_travel_times.json'}
FEATURE_COLUMNS_DEFAULT = ['travel_time_ij', 'same_zone', 'zone_changed', 'zone_missing_in_pair', 'number_of_stops', 'route_progress', 'remaining_stop_count', 'current_is_station', 'current_is_dropoff', 'candidate_is_station', 'candidate_is_dropoff', 'candidate_package_count', 'candidate_total_planned_service_time', 'candidate_has_time_window', 'candidate_time_window_package_count', 'candidate_total_package_volume_cm3', 'candidate_delivered_count', 'candidate_attempted_count', 'candidate_rejected_count', 'candidate_unknown_status_count']
WEIGHT_COLS = ['travel_weight', 'preference_weight', 'zone_weight', 'time_window_weight', 'workload_weight']
P1_WEIGHTS = dict(travel_weight=0.35, preference_weight=0.4, zone_weight=0.15, time_window_weight=0.05, workload_weight=0.05)
DIAGNOSTIC_WEIGHTS = [dict(travel_weight=1.0, preference_weight=0.0, zone_weight=0.0, time_window_weight=0.0, workload_weight=0.0), dict(travel_weight=0.0, preference_weight=1.0, zone_weight=0.0, time_window_weight=0.0, workload_weight=0.0), dict(travel_weight=0.25, preference_weight=0.25, zone_weight=0.4, time_window_weight=0.05, workload_weight=0.05), dict(travel_weight=0.2, preference_weight=0.2, zone_weight=0.2, time_window_weight=0.2, workload_weight=0.2)]
STOP_DEFAULTS = dict(zone=UNKNOWN_ZONE, type='', is_station=0, is_dropoff=0, package_count=0.0, total_planned_service_time=0.0, has_time_window=0, time_window_package_count=0.0, total_package_volume_cm3=0.0, delivered_count=0.0, attempted_count=0.0, rejected_count=0.0, unknown_status_count=0.0)
AVG_METRICS = ['actual_total_travel_time', 'generated_total_travel_time', 'travel_time_ratio_to_actual', 'lcs_similarity', 'position_match_ratio', 'generated_same_zone_ratio', 'actual_same_zone_ratio', 'zone_change_count']
WEIGHT_RESULT_COLUMNS = ['weight_id', *WEIGHT_COLS, 'route_count', 'avg_actual_total_travel_time', 'avg_generated_total_travel_time', 'avg_travel_time_ratio_to_actual', 'avg_lcs_similarity', 'avg_position_match_ratio', 'avg_generated_same_zone_ratio', 'avg_actual_same_zone_ratio', 'avg_zone_change_count', 'valid_route_rate', 'travel_score', 'lcs_score', 'same_zone_score', 'position_score', 'validation_score', 'rank']
ROUTE_METRIC_COLUMNS = ['route_id', 'method', 'weight_id', *WEIGHT_COLS, 'stop_count', 'actual_total_travel_time', 'generated_total_travel_time', 'travel_time_ratio_to_actual', 'lcs_similarity', 'position_match_ratio', 'generated_same_zone_ratio', 'actual_same_zone_ratio', 'zone_change_count', 'route_valid']

def parse_args():
    parser = argparse.ArgumentParser(description='Search hybrid route-generation weights using a trained preference model.')
    parser.add_argument('--config', type=Path, default=DEFAULT_CONFIG)
    parser.add_argument('--model-output-dir', type=Path, default=None)
    parser.add_argument('--model-path', type=Path, default=None)
    parser.add_argument('--feature-columns', type=Path, default=None)
    parser.add_argument('--output-dir', type=Path, default=None)
    parser.add_argument('--split', choices=VALID_SPLITS, default='validation')
    parser.add_argument('--max-routes', type=int, default=None)
    parser.add_argument('--route-ids', type=Path, default=None)
    parser.add_argument('--weight-step', type=float, default=0.1)
    parser.add_argument('--max-weight-combinations', type=int, default=None)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--save-route-level', action='store_true')
    parser.add_argument('--save-sequences', action='store_true')
    parser.add_argument('--overwrite', action='store_true')
    parser.add_argument('--verbose', action='store_true')
    return parser.parse_args()

def load_config(path):
    if not path.exists():
        warnings.warn(f'Config not found: {path}. Falling back to defaults.')
        return {}
    try:
        import yaml
    except ImportError as exc:
        raise SystemExit('Please install pyyaml: pip install pyyaml') from exc
    with path.open('r', encoding='utf-8') as file_obj:
        data = yaml.safe_load(file_obj) or {}
    if not isinstance(data, dict):
        raise ValueError(f'Config must be a YAML mapping: {path}')
    return data

def expand_path(value):
    path = Path(str(value)).expanduser()
    return path if path.is_absolute() else path.resolve()

def candidate_paths(entry, default=None):
    if isinstance(entry, dict) and 'candidates' in entry:
        values = list(entry['candidates'])
    elif isinstance(entry, (list, tuple)):
        values = list(entry)
    elif entry is None:
        values = []
    else:
        values = [entry]
    if default is not None:
        values.append(default)
    return [value for value in values if value is not None]

def choose_path(entry, default, prefer_existing=True):
    paths = [expand_path(value) for value in candidate_paths(entry, default)]
    if prefer_existing:
        for path in paths:
            if path.exists():
                return path
    if not paths:
        raise ValueError('No path candidates were available.')
    return paths[0]

def resolve_paths(args, config):
    paths_cfg = config.get('paths', {}) if isinstance(config.get('paths'), dict) else {}
    outputs_cfg = config.get('outputs', {}) if isinstance(config.get('outputs'), dict) else {}
    final_dir = choose_path(paths_cfg.get('final_cleaned_dir'), DEFAULT_FINAL, prefer_existing=False)
    processed_dir = choose_path(paths_cfg.get('processed_dir'), DEFAULT_PROCESSED, prefer_existing=False)
    model_dir = expand_path(args.model_output_dir or outputs_cfg.get('model_full_dir', DEFAULT_MODEL_DIR))
    route_id_defaults = {'train': final_dir / 'train_route_ids.csv', 'validation': final_dir / 'validation_route_ids.csv', 'test': final_dir / 'test_route_ids.csv'}
    route_id_keys = {'train': 'train_route_ids', 'validation': 'validation_route_ids', 'test': 'test_route_ids'}
    matrix_cfg = paths_cfg.get('travel_time_matrices', {})
    matrix_cfg = matrix_cfg if isinstance(matrix_cfg, dict) else {}
    transitions_default = [final_dir / 'actual_transition_travel_time_complete_routes.csv', final_dir / 'actual_transition_travel_time_clean.csv']
    return {'model_output_dir': model_dir, 'model_path': expand_path(args.model_path or model_dir / 'models/best_model.joblib'), 'feature_columns_path': expand_path(args.feature_columns or model_dir / 'feature_columns.json'), 'output_dir': expand_path(args.output_dir or outputs_cfg.get('hybrid_weight_search_dir', DEFAULT_OUTPUT_DIR)), 'route_ids': expand_path(args.route_ids) if args.route_ids else choose_path(paths_cfg.get(route_id_keys[args.split]), route_id_defaults[args.split]), 'route_splits': choose_path(paths_cfg.get('route_splits'), final_dir / 'route_splits.csv'), 'transitions': choose_path(paths_cfg.get('actual_transitions_with_travel_time', transitions_default), transitions_default[0]), 'stops_base_features': choose_path(paths_cfg.get('stops_base_features'), final_dir / 'stops_base_features.csv'), 'stop_package_features': choose_path(paths_cfg.get('stop_package_features'), final_dir / 'stop_package_features.csv'), 'routes_summary': choose_path(paths_cfg.get('routes_summary'), processed_dir / 'routes_summary.csv'), 'source_lookup': choose_path(paths_cfg.get('route_travel_time_source_lookup'), processed_dir / 'travel_time_multisource_outputs/route_travel_time_source_lookup.csv'), 'matrix_paths': {source: choose_path(matrix_cfg.get(source), DEFAULT_MATRICES[source], prefer_existing=False) for source in SOURCE_PRIORITY}}

def require_file(path, label):
    if not path.exists():
        raise FileNotFoundError(f'{label} not found: {path}')
    return path

def safe_missing(value):
    if PAIRWISE.is_missing(value):
        return True
    return isinstance(value, str) and value.strip().lower() in {'', 'nan', 'none', 'null'}

def load_feature_columns(model_dir, path):
    if not path.exists():
        warnings.warn(f'Feature columns file not found: {path}. Falling back to defaults.')
        return (list(FEATURE_COLUMNS_DEFAULT), 'default_feature_list')
    with path.open('r', encoding='utf-8') as file_obj:
        payload = json.load(file_obj)
    raw = payload.get('feature_columns') or payload.get('feature_columns_used') if isinstance(payload, dict) else payload
    if isinstance(raw, str):
        cols = [part.strip() for part in raw.split(',') if part.strip()]
    elif isinstance(raw, list):
        cols = [str(part).strip() for part in raw if str(part).strip()]
    else:
        raise ValueError(f'No feature column list found in {path}')
    cols = [col for col in dict.fromkeys(cols) if col != 'route_score']
    if not cols:
        raise ValueError(f'No usable feature columns found for model output directory: {model_dir}')
    return (cols, str(path))

def select_route_ids(paths, split, max_routes):
    route_ids = PAIRWISE.load_route_ids(paths['route_ids'], max_routes)
    split_rows = PAIRWISE.read_csv_rows(paths['route_splits'])
    split_lookup = {str(PAIRWISE.first_present(row, ('route_id',))).strip(): str(PAIRWISE.first_present(row, ('split',))).strip() for row in split_rows if not safe_missing(row.get('route_id'))}
    wrong_split = [route_id for route_id in route_ids if split_lookup.get(route_id) and split_lookup[route_id] != split]
    missing = [route_id for route_id in route_ids if route_id not in split_lookup]
    if wrong_split:
        warnings.warn(f'{len(wrong_split)} selected route IDs do not match split={split} in route_splits.csv.')
    if missing:
        warnings.warn(f'{len(missing)} selected route IDs are missing from route_splits.csv.')
    return route_ids

def source_candidates(row):
    raw = str(PAIRWISE.first_present(row, ('travel_time_source', 'source', 'source_label', 'matched_source', 'selected_source'), '')).strip().lower()
    if raw in SOURCE_PRIORITY:
        return [raw]
    if raw == 'multiple_sources':
        return list(SOURCE_PRIORITY)
    if raw == 'missing_source':
        return []
    found = []
    for source in SOURCE_PRIORITY:
        marker = str(PAIRWISE.first_present(row, (f'found_in_{source}', f'has_{source}', f'in_{source}', source), '')).strip().lower()
        if source in raw or marker in {'1', 'true', 'yes', 'y', source}:
            found.append(source)
    return list(dict.fromkeys(found))

def load_source_lookup(path, route_ids):
    out = {}
    for row in PAIRWISE.read_csv_rows(require_file(path, 'route travel-time source lookup')):
        route_id = str(PAIRWISE.first_present(row, ('route_id',))).strip()
        if route_id in route_ids:
            out[route_id] = source_candidates(row)
    return out

def stop_info(route_id, stop_id, stops, packages):
    info = dict(STOP_DEFAULTS)
    info.update(stops.get((route_id, stop_id), {}))
    info.update(packages.get((route_id, stop_id), {}))
    return info

def travel_time(matrix, from_stop, to_stop):
    return PAIRWISE.get_travel_time(matrix, from_stop, to_stop)

def pair_flags(current_zone, candidate_zone):
    missing = current_zone == UNKNOWN_ZONE or candidate_zone == UNKNOWN_ZONE
    return (int(not missing and current_zone == candidate_zone), int(not missing and current_zone != candidate_zone), int(missing))

def feature_row(route_id, current, candidate, ttime, position, seq_len, remaining, route_features, stops, packages):
    current_info = stop_info(route_id, current, stops, packages)
    candidate_info = stop_info(route_id, candidate, stops, packages)
    same, changed, missing = pair_flags(PAIRWISE.normalize_zone(current_info.get('zone')), PAIRWISE.normalize_zone(candidate_info.get('zone')))
    n_stops = int(route_features.get(route_id, {}).get('number_of_stops') or seq_len)
    denom = max(n_stops - 1, 1)
    return {'travel_time_ij': float(ttime) if ttime is not None else 0.0, 'same_zone': float(same), 'zone_changed': float(changed), 'zone_missing_in_pair': float(missing), 'number_of_stops': float(n_stops), 'route_progress': float(position / denom), 'remaining_stop_count': float(remaining), 'current_is_station': float(PAIRWISE.to_int(current_info.get('is_station', 0))), 'current_is_dropoff': float(PAIRWISE.to_int(current_info.get('is_dropoff', 0))), 'candidate_is_station': float(PAIRWISE.to_int(candidate_info.get('is_station', 0))), 'candidate_is_dropoff': float(PAIRWISE.to_int(candidate_info.get('is_dropoff', 0))), 'candidate_package_count': float(PAIRWISE.to_float(candidate_info.get('package_count', 0), 0) or 0), 'candidate_total_planned_service_time': float(PAIRWISE.to_float(candidate_info.get('total_planned_service_time', 0), 0) or 0), 'candidate_has_time_window': float(PAIRWISE.to_int(candidate_info.get('has_time_window', 0))), 'candidate_time_window_package_count': float(PAIRWISE.to_float(candidate_info.get('time_window_package_count', 0), 0) or 0), 'candidate_total_package_volume_cm3': float(PAIRWISE.to_float(candidate_info.get('total_package_volume_cm3', 0), 0) or 0), 'candidate_delivered_count': float(PAIRWISE.to_float(candidate_info.get('delivered_count', 0), 0) or 0), 'candidate_attempted_count': float(PAIRWISE.to_float(candidate_info.get('attempted_count', 0), 0) or 0), 'candidate_rejected_count': float(PAIRWISE.to_float(candidate_info.get('rejected_count', 0), 0) or 0), 'candidate_unknown_status_count': float(PAIRWISE.to_float(candidate_info.get('unknown_status_count', 0), 0) or 0)}

def feature_frame(rows, feature_columns):
    frame = pd.DataFrame(rows)
    for col in feature_columns:
        if col not in frame.columns:
            frame[col] = 0.0
    frame = frame.loc[:, list(feature_columns)].copy()
    for col in feature_columns:
        frame[col] = pd.to_numeric(frame[col], errors='coerce')
    return frame.fillna(0.0)

def predict_probability(model, rows, feature_columns):
    if not rows:
        return np.array([], dtype=float)
    x = feature_frame(rows, feature_columns)
    if hasattr(model, 'predict_proba'):
        proba = np.asarray(model.predict_proba(x), dtype=float)
        out = proba[:, 1] if proba.ndim == 2 and proba.shape[1] >= 2 else proba.reshape(-1)
    elif hasattr(model, 'decision_function'):
        scores = np.clip(np.asarray(model.decision_function(x), dtype=float).reshape(-1), -50, 50)
        out = 1.0 / (1.0 + np.exp(-scores))
    else:
        out = np.asarray(model.predict(x), dtype=float).reshape(-1)
    if len(out) != len(rows):
        raise ValueError(f'Model returned {len(out)} predictions for {len(rows)} candidates.')
    return np.clip(np.nan_to_num(out, nan=0.0, posinf=1.0, neginf=0.0), 0.0, 1.0)

def minmax(values, missing_value=0.0):
    values = np.asarray(values, dtype=float)
    out = np.full(values.shape, missing_value, dtype=float)
    mask = np.isfinite(values)
    if not mask.any():
        return out
    finite = values[mask]
    lo, hi = (float(np.min(finite)), float(np.max(finite)))
    out[mask] = 0.0 if math.isclose(lo, hi) else (finite - lo) / (hi - lo)
    return out

def workload(rows):
    return np.asarray([row.get('candidate_package_count', 0.0) + row.get('candidate_total_planned_service_time', 0.0) / 60.0 + row.get('candidate_time_window_package_count', 0.0) + row.get('candidate_total_package_volume_cm3', 0.0) / 10000.0 for row in rows], dtype=float)

def context_for(route_id, current, unvisited, position, seq, matrix, model, feature_columns, route_features, stops, packages):
    cand = list(unvisited)
    ttimes = [travel_time(matrix, current, stop) for stop in cand]
    travel = np.asarray([value if value is not None else np.nan for value in ttimes], dtype=float)
    rows = [feature_row(route_id, current, stop, value, position, len(seq), len(unvisited), route_features, stops, packages) for stop, value in zip(cand, ttimes)]
    proba = predict_probability(model, rows, feature_columns)
    zone_penalty = np.asarray([0.5 if row['zone_missing_in_pair'] == 1 else row['zone_changed'] for row in rows], dtype=float)
    time_window_risk = np.asarray([0.0 if row['candidate_has_time_window'] == 1 else 1.0 for row in rows], dtype=float)
    return {'candidates': cand, 'travel': travel, 'proba': proba, 'travel_norm': minmax(travel, missing_value=1.0), 'preference_penalty': 1.0 - proba, 'zone_penalty': zone_penalty, 'time_window_risk': time_window_risk, 'workload_penalty': minmax(workload(rows))}

def choose_next(method, ctx, weight):
    if method == 'travel_time_nearest_neighbor':
        values = np.where(np.isfinite(ctx['travel']), ctx['travel'], np.inf)
        return 0 if not np.isfinite(values).any() else int(np.argmin(values))
    if method == 'preference_greedy':
        return int(np.argmax(ctx['proba']))
    if method != 'hybrid_greedy':
        raise ValueError(f'Unsupported route generation method: {method}')
    cost = weight['travel_weight'] * ctx['travel_norm'] + weight['preference_weight'] * ctx['preference_penalty'] + weight['zone_weight'] * ctx['zone_penalty'] + weight['time_window_weight'] * ctx['time_window_risk'] + weight['workload_weight'] * ctx['workload_penalty']
    return int(np.argmin(cost))

def generate_route(route_id, method, seq, matrix, model, feature_columns, route_features, stops, packages, weight, cache=None):
    if len(seq) <= 1:
        return list(seq)
    current = str(seq[0])
    unvisited = [str(stop) for stop in seq[1:]]
    generated = [current]
    cache = cache if cache is not None else {}
    while unvisited:
        key = (current, tuple(unvisited))
        if key not in cache:
            cache[key] = context_for(route_id, current, unvisited, len(generated) - 1, seq, matrix, model, feature_columns, route_features, stops, packages)
        selected = cache[key]['candidates'][choose_next(method, cache[key], weight)]
        generated.append(selected)
        unvisited.remove(selected)
        current = selected
    return generated

def route_travel(sequence, matrix):
    total = 0.0
    for left, right in zip(sequence[:-1], sequence[1:]):
        value = travel_time(matrix, str(left), str(right))
        if value is None:
            return float('nan')
        total += value
    return total

def lcs_length(actual, generated):
    if len(set(actual)) == len(actual) and len(set(generated)) == len(generated):
        positions = {stop: idx for idx, stop in enumerate(actual)}
        mapped = [positions[stop] for stop in generated if stop in positions]
        tails = []
        for value in mapped:
            lo, hi = (0, len(tails))
            while lo < hi:
                mid = (lo + hi) // 2
                if tails[mid] < value:
                    lo = mid + 1
                else:
                    hi = mid
            if lo == len(tails):
                tails.append(value)
            else:
                tails[lo] = value
        return len(tails)
    previous = [0] * (len(generated) + 1)
    for actual_stop in actual:
        current = [0]
        for idx, generated_stop in enumerate(generated, 1):
            current.append(previous[idx - 1] + 1 if actual_stop == generated_stop else max(previous[idx], current[-1]))
        previous = current
    return previous[-1] if actual and generated else 0

def zone_metrics(sequence, route_id, stops, packages):
    if len(sequence) <= 1:
        return (float('nan'), 0)
    same = changed = total = 0
    for left, right in zip(sequence[:-1], sequence[1:]):
        left_zone = PAIRWISE.normalize_zone(stop_info(route_id, str(left), stops, packages).get('zone'))
        right_zone = PAIRWISE.normalize_zone(stop_info(route_id, str(right), stops, packages).get('zone'))
        total += 1
        same += int(left_zone != UNKNOWN_ZONE and left_zone == right_zone)
        changed += int(left_zone != UNKNOWN_ZONE and right_zone != UNKNOWN_ZONE and (left_zone != right_zone))
    return (same / total if total else float('nan'), changed)

def evaluate(route_id, method, actual, generated, matrix, stops, packages, weight=None):
    actual_time = route_travel(actual, matrix)
    generated_time = route_travel(generated, matrix)
    generated_same_zone, zone_changes = zone_metrics(generated, route_id, stops, packages)
    actual_same_zone, _ = zone_metrics(actual, route_id, stops, packages)
    ratio = generated_time / actual_time if math.isfinite(actual_time) and math.isfinite(generated_time) and (actual_time > 0) else float('nan')
    weight = weight or {}
    return {'route_id': route_id, 'method': method, 'weight_id': weight.get('weight_id', ''), 'travel_weight': weight.get('travel_weight', ''), 'preference_weight': weight.get('preference_weight', ''), 'zone_weight': weight.get('zone_weight', ''), 'time_window_weight': weight.get('time_window_weight', ''), 'workload_weight': weight.get('workload_weight', ''), 'stop_count': len(actual), 'actual_total_travel_time': actual_time, 'generated_total_travel_time': generated_time, 'travel_time_ratio_to_actual': ratio, 'lcs_similarity': lcs_length(actual, generated) / len(actual) if actual else float('nan'), 'position_match_ratio': sum((1 for a, g in zip(actual, generated) if a == g)) / len(actual) if actual else float('nan'), 'generated_same_zone_ratio': generated_same_zone, 'actual_same_zone_ratio': actual_same_zone, 'zone_change_count': zone_changes, 'route_valid': len(actual) == len(generated) and len(set(generated)) == len(generated) and (set(actual) == set(generated))}

def validate_step(step):
    if step <= 0 or step > 1:
        raise ValueError('--weight-step must be in (0, 1].')
    units = int(round(1.0 / step))
    if not math.isclose(units * step, 1.0, rel_tol=1e-09, abs_tol=1e-09):
        raise ValueError('--weight-step must evenly divide 1.0, for example 0.10 or 0.25.')
    return units

def full_grid(step):
    units = validate_step(step)
    out = []
    for a in range(units + 1):
        for b in range(units - a + 1):
            for c in range(units - a - b + 1):
                for d in range(units - a - b - c + 1):
                    e = units - a - b - c - d
                    out.append({name: round(value / units, 10) for name, value in zip(WEIGHT_COLS, [a, b, c, d, e])})
    return out

def wkey(weight):
    return tuple((round(float(weight[name]), 10) for name in WEIGHT_COLS))

def closest_weight(grid, target):
    return min(grid, key=lambda weight: (sum((abs(float(weight[name]) - target[name]) for name in WEIGHT_COLS)), max((abs(float(weight[name]) - target[name]) for name in WEIGHT_COLS)), wkey(weight)))

def weight_grid(step, max_combinations, seed):
    grid = full_grid(step)
    if max_combinations is None or max_combinations <= 0 or max_combinations >= len(grid):
        selected = list(grid)
    else:
        forced, seen = ([], set())
        for target in [P1_WEIGHTS, *DIAGNOSTIC_WEIGHTS]:
            weight = closest_weight(grid, target)
            if wkey(weight) not in seen:
                forced.append(weight)
                seen.add(wkey(weight))
        remaining = [weight for weight in grid if wkey(weight) not in seen]
        keep = max_combinations - len(forced)
        if keep < 0:
            warnings.warn('--max-weight-combinations is smaller than required diagnostic weights.')
            selected = forced
        else:
            selected = forced + random.Random(seed).sample(remaining, min(keep, len(remaining)))
    return [dict(weight, weight_id=f'w{idx:06d}') for idx, weight in enumerate(selected, 1)]

def add_metric(acc, row):
    acc['route_count'] += 1
    acc['valid_route_sum'] += float(bool(row.get('route_valid')))
    for metric in AVG_METRICS:
        try:
            value = float(row.get(metric))
        except (TypeError, ValueError):
            continue
        if math.isfinite(value):
            acc['sum'][metric] += value
            acc['count'][metric] += 1

def acc_row(weight, acc):
    row = {'weight_id': weight['weight_id'], **{name: weight[name] for name in WEIGHT_COLS}, 'route_count': acc['route_count']}
    for metric in AVG_METRICS:
        count = acc['count'].get(metric, 0)
        row[f'avg_{metric}'] = acc['sum'][metric] / count if count else float('nan')
    row['valid_route_rate'] = acc['valid_route_sum'] / acc['route_count'] if acc['route_count'] else float('nan')
    return row

def score(values, higher):
    arr = pd.to_numeric(pd.Series(values), errors='coerce').to_numpy(float)
    out = np.zeros(arr.shape)
    mask = np.isfinite(arr)
    if not mask.any():
        return out
    finite = arr[mask]
    lo, hi = (float(np.min(finite)), float(np.max(finite)))
    out[mask] = 1.0 if math.isclose(lo, hi) else (finite - lo) / (hi - lo) if higher else 1.0 - (finite - lo) / (hi - lo)
    return out

def finalize_results(accumulators, weights):
    by_id = {weight['weight_id']: weight for weight in weights}
    df = pd.DataFrame([acc_row(by_id[weight_id], acc) for weight_id, acc in accumulators.items()])
    for col in WEIGHT_RESULT_COLUMNS:
        if col not in df.columns:
            df[col] = np.nan
    df['travel_score'] = score(df['avg_travel_time_ratio_to_actual'], higher=False)
    df['lcs_score'] = score(df['avg_lcs_similarity'], higher=True)
    df['same_zone_score'] = score(df['avg_generated_same_zone_ratio'], higher=True)
    df['position_score'] = score(df['avg_position_match_ratio'], higher=True)
    df['validation_score'] = 0.5 * df['travel_score'] + 0.25 * df['lcs_score'] + 0.2 * df['same_zone_score'] + 0.05 * df['position_score']
    df = df.sort_values(['validation_score', 'avg_travel_time_ratio_to_actual', 'avg_lcs_similarity', 'avg_generated_same_zone_ratio'], ascending=[False, True, False, False], kind='mergesort').reset_index(drop=True)
    df['rank'] = np.arange(1, len(df) + 1)
    return df.loc[:, WEIGHT_RESULT_COLUMNS]

class CsvWriter:

    def __init__(self, path, columns):
        self.file_obj = path.open('w', encoding='utf-8', newline='')
        self.columns = list(columns)
        self.writer = csv.DictWriter(self.file_obj, fieldnames=self.columns)
        self.writer.writeheader()

    def write(self, row):
        self.writer.writerow({column: row.get(column, '') for column in self.columns})

    def close(self):
        self.file_obj.close()

def stream_matrices(route_ids, route_sources, matrix_paths, handler, verbose):
    unresolved = set(route_ids)
    processed, seen = ([], set())
    for source in SOURCE_PRIORITY:
        needed = {route_id for route_id in unresolved if source in route_sources.get(route_id, [])}
        if not needed:
            continue
        path = matrix_paths[source]
        if not path.exists():
            warnings.warn(f'Travel-time matrix file not found for {source}: {path}')
            continue
        print(f'Streaming {len(needed)} route matrix/matrices from {source}: {path}')
        matched = 0
        for route_id, matrix in stream_top_level_object(path):
            route_id = str(route_id)
            if route_id not in needed or route_id not in unresolved:
                continue
            handler(route_id, matrix, source)
            unresolved.remove(route_id)
            processed.append(route_id)
            seen.add(route_id)
            matched += 1
            if verbose and matched % 25 == 0:
                print(f'  {source}: processed {matched} route(s)')
            if not needed & unresolved:
                break
        print(f'  {source}: processed {matched} route(s)')
    return (processed, [route_id for route_id in route_ids if route_id not in seen])

def write_df(path, df, columns=None):
    if columns is not None:
        for col in columns:
            if col not in df.columns:
                df[col] = np.nan
        df = df.loc[:, list(columns)]
    df.to_csv(path, index=False)

def clean_json(value):
    if isinstance(value, dict):
        return {str(key): clean_json(item) for key, item in value.items()}
    if isinstance(value, list):
        return [clean_json(item) for item in value]
    if isinstance(value, tuple):
        return [clean_json(item) for item in value]
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        value = float(value)
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, np.ndarray):
        return clean_json(value.tolist())
    if isinstance(value, np.bool_):
        return bool(value)
    return value

def write_json(path, payload):
    with path.open('w', encoding='utf-8') as file_obj:
        json.dump(clean_json(payload), file_obj, indent=2)
        file_obj.write('\n')

def prepare_output(output_dir, overwrite):
    if output_dir.exists() and any(output_dir.iterdir()):
        if not overwrite:
            raise FileExistsError(f'Output directory is not empty: {output_dir}. Use --overwrite to replace it.')
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

def run_weight_pass(route_ids, route_sources, paths, sequences, model, feature_columns, route_features, stops, packages, weights, save_route_level, verbose):
    accumulators = {weight['weight_id']: {'route_count': 0, 'valid_route_sum': 0.0, 'sum': defaultdict(float), 'count': defaultdict(int)} for weight in weights}
    writer = CsvWriter(paths['output_dir'] / 'route_metrics_all_weights.csv', ROUTE_METRIC_COLUMNS) if save_route_level else None

    def handle(route_id, matrix, source):
        if verbose:
            print(f'Optimizing weights for route {route_id} from {source}')
        cache = {}
        for weight in weights:
            generated = generate_route(route_id, 'hybrid_greedy', sequences[route_id], matrix, model, feature_columns, route_features, stops, packages, weight, cache)
            row = evaluate(route_id, 'hybrid_greedy', sequences[route_id], generated, matrix, stops, packages, weight)
            add_metric(accumulators[weight['weight_id']], row)
            if writer is not None:
                writer.write(row)
    try:
        processed, missing = stream_matrices(route_ids, route_sources, paths['matrix_paths'], handle, verbose)
    finally:
        if writer is not None:
            writer.close()
    if not processed:
        raise RuntimeError('No routes could be processed for hybrid weight search.')
    return (processed, missing, finalize_results(accumulators, weights))

def write_sequence(writer, route_id, method, sequence):
    for pos, stop_id in enumerate(sequence):
        writer.write({'route_id': route_id, 'method': method, 'position': pos, 'stop_id': stop_id})

def run_baseline_pass(route_ids, route_sources, paths, sequences, model, feature_columns, route_features, stops, packages, best_weight, save_sequences, verbose):
    baseline_rows, best_rows = ([], [])
    sequence_writer = CsvWriter(paths['output_dir'] / 'generated_sequences_best_weight.csv', ['route_id', 'method', 'position', 'stop_id']) if save_sequences else None

    def run_method(route_id, matrix, cache, method, label, weight):
        method_weight = weight or {col: '' for col in ['weight_id', *WEIGHT_COLS]}
        generated = generate_route(route_id, method, sequences[route_id], matrix, model, feature_columns, route_features, stops, packages, method_weight, cache)
        if sequence_writer is not None:
            write_sequence(sequence_writer, route_id, label, generated)
        return evaluate(route_id, label, sequences[route_id], generated, matrix, stops, packages, weight if label == 'hybrid_greedy_best_weight' else None)

    def handle(route_id, matrix, source):
        if verbose:
            print(f'Evaluating baselines for route {route_id} from {source}')
        cache = {}
        baseline_rows.append(run_method(route_id, matrix, cache, 'travel_time_nearest_neighbor', 'travel_time_nearest_neighbor', None))
        baseline_rows.append(run_method(route_id, matrix, cache, 'preference_greedy', 'preference_greedy', None))
        hybrid = run_method(route_id, matrix, cache, 'hybrid_greedy', 'hybrid_greedy_best_weight', best_weight)
        baseline_rows.append(hybrid)
        best_rows.append(hybrid)
    try:
        _processed, missing = stream_matrices(route_ids, route_sources, paths['matrix_paths'], handle, verbose)
        if missing:
            warnings.warn(f'{len(missing)} routes from the weight pass were missing in baseline pass.')
    finally:
        if sequence_writer is not None:
            sequence_writer.close()
    return (pd.DataFrame(best_rows), pd.DataFrame(baseline_rows))

def summarize_methods(rows):
    if rows.empty:
        return pd.DataFrame()
    return rows.groupby('method', dropna=False).agg(route_count=('route_id', 'count'), avg_actual_total_travel_time=('actual_total_travel_time', 'mean'), avg_generated_total_travel_time=('generated_total_travel_time', 'mean'), avg_travel_time_ratio_to_actual=('travel_time_ratio_to_actual', 'mean'), avg_lcs_similarity=('lcs_similarity', 'mean'), avg_position_match_ratio=('position_match_ratio', 'mean'), avg_generated_same_zone_ratio=('generated_same_zone_ratio', 'mean'), avg_actual_same_zone_ratio=('actual_same_zone_ratio', 'mean'), avg_zone_change_count=('zone_change_count', 'mean'), valid_route_rate=('route_valid', 'mean')).reset_index()

def print_paths(paths):
    print('Runtime inputs:')
    for key in ['model_path', 'feature_columns_path', 'route_splits', 'route_ids', 'transitions', 'stops_base_features', 'stop_package_features', 'routes_summary', 'source_lookup', 'output_dir']:
        print(f'  {key}: {paths[key]}')
    for source, path in paths['matrix_paths'].items():
        print(f'  matrix_{source}: {path}')

def main():
    args = parse_args()
    config = load_config(args.config)
    paths = resolve_paths(args, config)
    print_paths(paths)
    require_file(paths['model_path'], 'best model')
    model = joblib.load(paths['model_path'])
    feature_columns, feature_source = load_feature_columns(paths['model_output_dir'], paths['feature_columns_path'])
    print(f"Loaded model: {paths['model_path']}")
    print(f'Feature columns ({len(feature_columns)}): {feature_columns}')
    route_ids = select_route_ids(paths, args.split, args.max_routes)
    if not route_ids:
        raise ValueError(f'No route IDs selected for split={args.split}.')
    print(f'Selected {len(route_ids)} route(s) for split={args.split}')
    requested = set(route_ids)
    route_features = PAIRWISE.load_route_features(paths['routes_summary'])
    stops = PAIRWISE.load_stop_features(paths['stops_base_features'], requested)
    packages = PAIRWISE.load_package_features(paths['stop_package_features'], requested)
    transitions = PAIRWISE.load_transitions(paths['transitions'], requested)
    source_lookup = load_source_lookup(paths['source_lookup'], requested)
    sequences, invalid_sequences = ({}, {})
    for route_id in route_ids:
        seq = PAIRWISE.build_sequence(transitions.get(route_id, []))
        if len(seq) < 2:
            invalid_sequences[route_id] = 'missing_or_short_sequence'
        elif len(set(seq)) != len(seq):
            invalid_sequences[route_id] = 'duplicate_stop_in_sequence'
        else:
            sequences[route_id] = seq
    valid_route_ids = [route_id for route_id in route_ids if route_id in sequences]
    route_sources, missing_sources = ({}, [])
    for route_id in valid_route_ids:
        srcs = source_lookup.get(route_id, [])
        if srcs:
            route_sources[route_id] = srcs
        else:
            missing_sources.append(route_id)
    processable = [route_id for route_id in valid_route_ids if route_id not in set(missing_sources)]
    weights = weight_grid(args.weight_step, args.max_weight_combinations, args.seed)
    print(f'Testing {len(weights)} weight combination(s) with step={args.weight_step}')
    prepare_output(paths['output_dir'], args.overwrite)
    processed, missing_matrices, results = run_weight_pass(processable, route_sources, paths, sequences, model, feature_columns, route_features, stops, packages, weights, args.save_route_level, args.verbose)
    write_df(paths['output_dir'] / 'weight_grid_results.csv', results, WEIGHT_RESULT_COLUMNS)
    best = results.iloc[0].to_dict()
    best_weight = {name: float(best[name]) for name in WEIGHT_COLS}
    best_weight['weight_id'] = str(best['weight_id'])
    write_df(paths['output_dir'] / 'best_weight_summary.csv', pd.DataFrame([best]))
    write_json(paths['output_dir'] / 'best_weight_summary.json', best)
    best_route_metrics, baseline_route_metrics = run_baseline_pass(processed, route_sources, paths, sequences, model, feature_columns, route_features, stops, packages, best_weight, args.save_sequences, args.verbose)
    write_df(paths['output_dir'] / 'best_weight_route_metrics.csv', best_route_metrics, ROUTE_METRIC_COLUMNS)
    write_df(paths['output_dir'] / 'baseline_route_metrics.csv', baseline_route_metrics, ROUTE_METRIC_COLUMNS)
    write_df(paths['output_dir'] / 'baseline_method_summary.csv', summarize_methods(baseline_route_metrics))
    run_summary = {'run_timestamp': datetime.now(timezone.utc).isoformat(), 'split': args.split, 'route_count_requested': len(route_ids), 'route_count_processed': len(processed), 'model_path': str(paths['model_path']), 'feature_columns_source': feature_source, 'weight_step': args.weight_step, 'weight_combinations_tested': len(weights), 'max_weight_combinations': args.max_weight_combinations, 'seed': args.seed, 'output_dir': str(paths['output_dir']), 'routes_skipped_missing_source': len(missing_sources), 'routes_skipped_missing_matrix': len(missing_matrices), 'routes_skipped_invalid_sequence': len(invalid_sequences)}
    write_df(paths['output_dir'] / 'hybrid_weight_search_run_summary.csv', pd.DataFrame([run_summary]))
    write_json(paths['output_dir'] / 'hybrid_weight_search_run_summary.json', run_summary)
    print('\nHybrid weight search complete.')
    print(f'Routes requested: {len(route_ids)}')
    print(f'Routes processed: {len(processed)}')
    print(f'Skipped missing source: {len(missing_sources)}')
    print(f'Skipped missing matrix: {len(missing_matrices)}')
    print(f'Skipped invalid sequence: {len(invalid_sequences)}')
    print(f"Best weight_id: {best_weight['weight_id']}")
    print(f"Best weights: travel={best_weight['travel_weight']:.4f}, preference={best_weight['preference_weight']:.4f}, zone={best_weight['zone_weight']:.4f}, time_window={best_weight['time_window_weight']:.4f}, workload={best_weight['workload_weight']:.4f}")
    print(f"Weight results: {paths['output_dir'] / 'weight_grid_results.csv'}")
    print(f"Output directory: {paths['output_dir']}")
    return 0
if __name__ == '__main__':
    raise SystemExit(main())
