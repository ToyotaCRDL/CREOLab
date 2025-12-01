"""
Chart generation functions for procedure evaluation pipeline.
"""

import os
import json
import re
from datetime import datetime
from statistics import mean
from collections import defaultdict

# Visualization libraries
import matplotlib.pyplot as plt
import numpy as np

# Local helper modules
from io_utils import save_text
from chart_helpers import (
    apply_chart_style, save_chart_with_error_handling, save_chart_data_json,
    get_rubric_category_order, get_category_visual_style,
    DEFAULT_DEDUCTION_STYLE, DEDUCTION_BAR_ALPHA
)
from stats_helpers import mean_and_95ci

# Constants for procedure types and labels
PROC_MANUAL = "manual_object_detection"
PROC_AUTO = "auto_object_detection"

PROCEDURE_MAPPING = {
    PROC_MANUAL: "Manual Object Detection",
    PROC_AUTO: "Auto Object Detection"
}

PROCEDURE_ORDER = [PROC_MANUAL, PROC_AUTO]
PROCEDURE_LABELS = [PROCEDURE_MAPPING[k] for k in PROCEDURE_ORDER]
INTEGRATION_TYPES = PROCEDURE_ORDER

RUBRIC_EVALUATION_RESULTS_FILE = "rubric_evaluation_results.json"


def load_evaluation_scores_from_file(rubric_results_file):
    """
    Load evaluation scores and deductions from rubric evaluation results file.
    
    Args:
        rubric_results_file: Path to rubric_evaluation_results.json
        
    Returns:
        Tuple of (scores_dict, deductions_dict) where:
        - scores_dict: {procedure_name: final_score}
        - deductions_dict: {procedure_name: {category: total_points}}
        Returns (None, None) if file cannot be loaded or has invalid structure.
    """
    
    with open(rubric_results_file, 'r', encoding='utf-8') as f:
        rubric_data = json.load(f)
        
    evaluations = rubric_data["evaluations"]
    scores = {}
    deductions = {}
    
    for method_key in PROCEDURE_ORDER:
        if method_key in evaluations:
            eval_data = evaluations[method_key]
            if "final_score" in eval_data:
                proc_name = PROCEDURE_MAPPING[method_key]
                scores[proc_name] = eval_data["final_score"]
                
                if "category_summary" in eval_data:
                    proc_deductions = {}
                    for category, category_data in eval_data["category_summary"].items():
                        if "total_points" in category_data:
                            proc_deductions[category] = category_data["total_points"]
                    deductions[proc_name] = proc_deductions
    
    return scores if scores else None, deductions if deductions else None

def parse_take_name(take_name):
    """
    Parse take name to extract scenario and decoy numbers.
    
    Args:
        take_name: String like 'scenario05_decoy0'
        
    Returns:
        Tuple of (scenario_number, decoy_number) as integers, or (None, None) if parsing fails
    """
    match = re.match(r'scenario(\d+)_decoy(\d+)', take_name)
    if match:
        return int(match.group(1)), int(match.group(2))
    return None, None


def create_iteration_bar_chart(take_dir, take_name):
    """Create bar chart for all iterations of a take."""
    procedure_labels = PROCEDURE_LABELS
    
    # Collect scores and deductions from all iterations
    iteration_data = []
    
    for iter_name in sorted(os.listdir(take_dir)):
        if iter_name.startswith("iter_"):
            iter_dir = os.path.join(take_dir, iter_name)
            rubric_results_file = os.path.join(iter_dir, "evaluation", RUBRIC_EVALUATION_RESULTS_FILE)
            
            scores, deductions = load_evaluation_scores_from_file(rubric_results_file)
            if scores:
                iteration_data.append({
                    'label': iter_name.replace("iter_", "Iter "),
                    'scores': scores,
                    'deductions': deductions if deductions else {}
                })
    
    if not iteration_data:
        print(f"  No iteration scores found for {take_name}")
        return
    
    # Create chart for each iteration
    for i, data in enumerate(iteration_data):
        # Determine output directory
        iter_num = i + 1
        iter_dir = os.path.join(take_dir, f"iter_{iter_num:02d}")
        eval_dir = os.path.join(iter_dir, "evaluation")
        
        chart_path = os.path.join(eval_dir, "evaluation_bar_chart.png")
        title = f"{take_name} - {data['label']}"
        
        create_deduction_breakdown_bar_chart(
            data['scores'], procedure_labels, chart_path, title, 
            None, data['deductions'] if data['deductions'] else None
        )


def compute_take_aggregate_results(take_dir, take_name, exp_logger):
    """Compute aggregate results across all iterations for a take."""
    procedure_labels = PROCEDURE_LABELS
    
    # Collect all evaluation results
    all_scores = []
    all_deductions = []
    
    for iter_name in os.listdir(take_dir):
        if iter_name.startswith("iter_"):
            iter_dir = os.path.join(take_dir, iter_name)
            rubric_results_file = os.path.join(iter_dir, "evaluation", RUBRIC_EVALUATION_RESULTS_FILE)
            
            scores, deductions = load_evaluation_scores_from_file(rubric_results_file)
            if scores:
                all_scores.append(scores)
                if deductions:
                    all_deductions.append(deductions)
    
    if not all_scores:
        print(f"  No evaluation scores found for {take_name}")
        return
    
    # Calculate average scores and raw data for error bars
    avg_scores = {}
    raw_data = {}
    for proc_name in procedure_labels:
        proc_scores = [s[proc_name] for s in all_scores if proc_name in s]
        if proc_scores:
            avg_scores[proc_name] = mean(proc_scores)
            raw_data[proc_name] = [[score] for score in proc_scores]
        else:
            avg_scores[proc_name] = 0
            raw_data[proc_name] = []
    
    # Calculate average deductions
    avg_deductions = {}
    if all_deductions:
        for proc_name in procedure_labels:
            proc_deductions = {}
            for deduction_dict in all_deductions:
                if proc_name in deduction_dict:
                    for category, points in deduction_dict[proc_name].items():
                        if category not in proc_deductions:
                            proc_deductions[category] = []
                        proc_deductions[category].append(points)
            
            avg_proc_deductions = {}
            for category, points_list in proc_deductions.items():
                if points_list:
                    avg_proc_deductions[category] = mean(points_list)
            
            if avg_proc_deductions:
                avg_deductions[proc_name] = avg_proc_deductions
            
    # Create aggregate directory
    aggregate_dir = exp_logger.get_subdir("aggregate") if exp_logger else os.path.join(take_dir, "aggregate")
    if not exp_logger:
        os.makedirs(aggregate_dir, exist_ok=True)
    
    # Save aggregate results
    aggregate_results = {
        "take_name": take_name,
            "total_iterations": len(all_scores),
        "average_scores": avg_scores,
        "average_deductions": avg_deductions,
            "timestamp": datetime.now().isoformat()
    }
    
    aggregate_results_file = os.path.join(aggregate_dir, "aggregate_results.json")
    with open(aggregate_results_file, 'w', encoding='utf-8') as f:
        json.dump(aggregate_results, f, indent=2, ensure_ascii=False)
    
    # Create aggregate summary text
    summary_lines = ["Take Aggregate Evaluation Summary", "=" * 50, ""]
    summary_lines.append(f"Take: {take_name}")
    summary_lines.append(f"Total iterations: {len(all_scores)}")
    summary_lines.append("")
    summary_lines.append("Average Scores:")
    for proc_name in procedure_labels:
        summary_lines.append(f"  {proc_name}: {avg_scores.get(proc_name, 0):.2f}")
    
    aggregate_summary_file = os.path.join(aggregate_dir, "aggregate_summary.txt")
    save_text(aggregate_summary_file, "\n".join(summary_lines))
    
    # Create aggregate bar chart
    chart_path = os.path.join(aggregate_dir, "take_aggregate_rubric_deduction_breakdown.png")
    title = f"{take_name} Aggregate ({len(all_scores)} iterations)"
    
    has_sufficient_data = any(len(data) > 1 for data in raw_data.values())
    create_deduction_breakdown_bar_chart(
        avg_scores, procedure_labels, chart_path, title,
        raw_data if has_sufficient_data else None,
        avg_deductions if avg_deductions else None
    )
    
    print(f"  Aggregate results saved: {aggregate_results_file}")
    print(f"  Aggregate summary saved: {aggregate_summary_file}")
    print(f"  Aggregate chart saved: {chart_path}")


def create_deduction_breakdown_bar_chart(scores_data, procedure_labels, save_path, title="Cross-Analysis Results", raw_data=None, deduction_data=None):
    """Create bar chart with deduction breakdown and optional error bars."""
    # Calculate overall scores (average across all criteria) for each procedure
    overall_scores = {}
    error_bars = {}
    
    for proc_name in procedure_labels:
        if proc_name in scores_data:
            # Handle both single values and lists
            if isinstance(scores_data[proc_name], (int, float)):
                # Single score value
                overall_scores[proc_name] = scores_data[proc_name]
            elif isinstance(scores_data[proc_name], list):
                # List of scores - calculate average
                overall_scores[proc_name] = mean(scores_data[proc_name])
            else:
                # Skip invalid data
                continue
            
            # Calculate error bars if raw data is available
            if raw_data and proc_name in raw_data and len(raw_data[proc_name]) > 1:
                # Calculate overall scores for each individual evaluation
                individual_overall_scores = []
                for individual_scores in raw_data[proc_name]:
                    if len(individual_scores) > 0:
                        individual_overall_scores.append(mean(individual_scores))
                
                individual_scores_clean = [s for s in individual_overall_scores if np.isfinite(s)]
                _, ci = mean_and_95ci(individual_scores_clean)
                error_bars[proc_name] = ci
            else:
                error_bars[proc_name] = 0.0
    
    if not overall_scores:
        print(f"No valid data for cross-analysis bar chart")
        return
    
    # Check if we have deduction data to create stacked chart
    # Ensure consistent order: Auto, Manual
    ordered_names = []
    for label in procedure_labels:
        if label in overall_scores:
            ordered_names.append(label)
    
    names = ordered_names
    values = [overall_scores[name] for name in names]
    errors = [error_bars[name] for name in names]
    
    if deduction_data:
        # Create stacked bar chart with deduction breakdown
        fig, ax = plt.subplots(figsize=(12, 8))
        
        # Get all unique deduction categories in rubric order
        all_categories = set()
        for proc_name in names:
            if proc_name in deduction_data:
                all_categories.update(deduction_data[proc_name].keys())
        
        # Order categories according to rubric order (1-6)
        standard_order = get_rubric_category_order()
        categories = []
        for category in standard_order:
            if category in all_categories:
                categories.append(category)
        # Add any additional categories not in standard order
        for category in sorted(all_categories):
            if category not in categories:
                categories.append(category)
        
        # Prepare deduction data for each procedure
        deductions_by_proc = {}
        for proc_name in names:
            deductions_by_proc[proc_name] = []
            for category in categories:
                if proc_name in deduction_data and category in deduction_data[proc_name]:
                    deductions_by_proc[proc_name].append(deduction_data[proc_name][category])
                else:
                    deductions_by_proc[proc_name].append(0)
        
        # Create stacked bars
        x_pos = np.arange(len(names))
        width = 0.6
        
        # Final scores (red, bottom)
        bars_final = ax.bar(x_pos, values, width, color='red', alpha=0.8, 
                           yerr=errors, capsize=5, error_kw={'color': 'darkred', 'linewidth': 2},
                           label='Final Score')
        
        # Deduction categories (colorblind-friendly with patterns, stacked on top)
        style_mapping = get_category_visual_style()
        bottom_values = values[:]
        
        for i, category in enumerate(categories):
            deduction_values = [deductions_by_proc[proc_name][i] for proc_name in names]
            if any(d > 0 for d in deduction_values):
                style = style_mapping.get(category, DEFAULT_DEDUCTION_STYLE)
                ax.bar(x_pos, deduction_values, width, bottom=bottom_values,
                      color=style['color'], alpha=DEDUCTION_BAR_ALPHA,
                      hatch=style['pattern'], edgecolor=style['edgecolor'], linewidth=style['linewidth'],
                      label=f'{category}')
                bottom_values = [b + d for b, d in zip(bottom_values, deduction_values)]
        
        # Chart settings with 100-point reference line
        apply_chart_style(
            ax, 
            title=f'{title}\n(Red: Final Score with 95% CI, Patterns: Deduction Categories)',
            ylabel='Points',
            xlabel='Procedure',
            ylim=(0, 105),
            xticks=x_pos,
            xticklabels=names,
            add_reference_line=True,
            reference_x_pos=len(names)-0.5
        )
        
        # Add value labels on final score bars
        for i, (final_score, error) in enumerate(zip(values, errors)):
            ax.text(x_pos[i], final_score/2, f'{final_score:.1f}',
                   ha='center', va='center', fontweight='bold',
                   color='white', fontsize=11)
        
        # Add legend with reversed order to match visual stacking
        handles, labels = ax.get_legend_handles_labels()
        ax.legend(handles[::-1], labels[::-1], loc='center left', bbox_to_anchor=(1, 0.5))
        
    else:
        # Create simple bar chart without deduction breakdown
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Use procedure-specific colors (Red, Green, Blue)
        procedure_colors = ['#D73027', '#1A9850', '#313695']
        bars = ax.bar(names, values, yerr=errors, color=procedure_colors[:len(names)], alpha=0.8, 
                     edgecolor='black', linewidth=1, capsize=5, error_kw={'linewidth': 2})
        
        # Customize the chart
        apply_chart_style(
            ax,
            title=title,
            ylabel='Overall Score (Average Across All Criteria)',
            ylim=(0, 100)
        )
        
        # Add value labels on top of bars (above error bars)
        for bar, value, error in zip(bars, values, errors):
            height = bar.get_height() + error
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.05,
                   f'{value:.3f}', ha='center', va='bottom', 
                   fontsize=11, fontweight='bold')
        
        # Rotate x-axis labels if needed
        plt.xticks(rotation=45, ha='right')
    
    chart_path = save_chart_with_error_handling(save_path, "Cross-analysis bar chart")
    
    # Save chart data as JSON
    if chart_path:
        chart_data = {
            "title": title,
            "procedure_labels": names,
            "overall_scores": dict(zip(names, values)),
            "error_bars": dict(zip(names, errors))
        }
        if deduction_data:
            chart_data["deduction_breakdown"] = deduction_data
        if raw_data:
            chart_data["raw_data"] = {k: v for k, v in raw_data.items() if k in names}
        save_chart_data_json(chart_path, chart_data)


def create_method_decoy_progression_chart(decoy_data, method_name, save_path, deduction_data=None):
    """Create a line chart showing score progression across decoy numbers for a specific method."""
    if not decoy_data:
        print(f"No data available for {method_name} decoy progression")
        return
    
    # Sort decoy numbers and calculate statistics
    decoy_numbers = sorted([int(k) for k in decoy_data.keys()])
    mean_scores = []
    error_bars = []
    
    for decoy_num in decoy_numbers:
        scores = decoy_data[str(decoy_num)]
        if scores:
            mean_score, ci = mean_and_95ci(scores)
            mean_scores.append(mean_score)
            error_bars.append(ci)
        else:
            mean_scores.append(0)
            error_bars.append(0.0)
    
    # Check if we have deduction data to create stacked area chart
    if deduction_data:
        # Create stacked area chart with deduction breakdown
        fig, ax = plt.subplots(figsize=(12, 8))
        
        # Get all unique deduction categories in rubric order
        all_categories = set()
        for decoy_num in decoy_numbers:
            if str(decoy_num) in deduction_data:
                all_categories.update(deduction_data[str(decoy_num)].keys())
        
        # Order categories according to rubric order (1-6)
        standard_order = get_rubric_category_order()
        categories = []
        for category in standard_order:
            if category in all_categories:
                categories.append(category)
        # Add any additional categories not in standard order
        for category in sorted(all_categories):
            if category not in categories:
                categories.append(category)
        
        # Prepare deduction data for each decoy number
        deduction_means = {category: [] for category in categories}
        
        for decoy_num in decoy_numbers:
            for category in categories:
                if str(decoy_num) in deduction_data and category in deduction_data[str(decoy_num)]:
                    deduction_means[category].append(deduction_data[str(decoy_num)][category])
                else:
                    deduction_means[category].append(0)
        
        # Create stacked area chart
        # Final scores (red line with error bars)
        ax.errorbar(decoy_numbers, mean_scores, yerr=error_bars,
                   marker='o', linewidth=3, markersize=8, capsize=5,
                   color='red', ecolor='darkred', alpha=0.8, 
                   label='Final Score', zorder=10, elinewidth=2)
        
        # Deduction categories (stacked areas with colorblind-friendly patterns)
        style_mapping = get_category_visual_style()
        bottom_values = mean_scores[:]
        
        for i, category in enumerate(categories):
            deduction_values = deduction_means[category]
            if any(d > 0 for d in deduction_values):
                style = style_mapping.get(category, DEFAULT_DEDUCTION_STYLE)
                ax.fill_between(decoy_numbers, bottom_values, 
                               [b + d for b, d in zip(bottom_values, deduction_values)],
                               color=style['color'], alpha=0.7,
                               hatch=style['pattern'], edgecolor=style['edgecolor'], linewidth=style['linewidth'],
                               label=f'{category}')
                bottom_values = [b + d for b, d in zip(bottom_values, deduction_values)]
        
        # Chart settings with 100-point reference line
        apply_chart_style(
            ax,
            title=f'{method_name} - Score Progression with Deduction Breakdown\n(Red: Final Score with Error Bars, Patterns: Deduction Categories)',
            ylabel='Points',
            xlabel='Decoy Number',
            ylim=(0, 105),
            xticks=decoy_numbers,
            add_reference_line=True,
            reference_x_pos=decoy_numbers[-1]
        )
        
        # Add legend with reversed order to match visual stacking
        handles, labels = ax.get_legend_handles_labels()
        ax.legend(handles[::-1], labels[::-1], loc='center left', bbox_to_anchor=(1, 0.5))
        
    else:
        # Create simple line chart without deduction breakdown
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Plot line with error bars
        ax.errorbar(decoy_numbers, mean_scores, yerr=error_bars, 
                    marker='o', linewidth=2, markersize=8, capsize=5,
                    color='#1f77b4', ecolor='gray', alpha=0.8)
        
        # Customize the chart
        apply_chart_style(
            ax,
            title=f'{method_name} - Score Progression Across Decoy Numbers',
            ylabel='Average Score',
            xlabel='Decoy Number',
            ylim=(0, 100),
            xticks=decoy_numbers
        )
    
    chart_path = save_chart_with_error_handling(save_path, "Method decoy progression chart")
    
    # Save chart data as JSON
    if chart_path:
        chart_data = {
            "method_name": method_name,
            "decoy_numbers": decoy_numbers,
            "mean_scores": mean_scores,
            "error_bars": error_bars
        }
        if deduction_data:
            # Convert deduction_data keys to strings for JSON serialization
            chart_data["deduction_breakdown"] = {str(k): v for k, v in deduction_data.items()}
        save_chart_data_json(chart_path, chart_data)

    
def create_cross_analysis_charts(batch_experiment_dir, split_name):
    """Create bar charts for cross-analysis using tidy data approach."""
    procedure_labels = PROCEDURE_LABELS
    
    # Phase 1: Collect all evaluations as tidy records
    # Record format: {take, scenario, decoy, iteration, procedure, score, deductions}
    records = []
    
    if not os.path.exists(batch_experiment_dir):
        print(f"Split directory not found: {batch_experiment_dir}")
        return
    
    for take_name in os.listdir(batch_experiment_dir):
        take_dir = os.path.join(batch_experiment_dir, take_name)
        if not os.path.isdir(take_dir):
            continue
        
        scenario_number, decoy_number = parse_take_name(take_name)
        if scenario_number is None or decoy_number is None:
            continue
        
        # Collect all iterations for this take
        for iter_name in os.listdir(take_dir):
            if not iter_name.startswith("iter_"):
                continue
                
            iter_dir = os.path.join(take_dir, iter_name)
            rubric_results_file = os.path.join(iter_dir, "evaluation", RUBRIC_EVALUATION_RESULTS_FILE)
            
            scores, deductions_dict = load_evaluation_scores_from_file(rubric_results_file)
            if not scores:
                continue
            
            # Create one record per procedure
            for proc_name in PROCEDURE_LABELS:
                if proc_name in scores:
                    proc_deductions = deductions_dict.get(proc_name, {}) if deductions_dict else {}
                    records.append({
                        'take': take_name,
                        'scenario': scenario_number,
                        'decoy': decoy_number,
                        'iteration': iter_name,
                        'procedure': proc_name,
                        'score': scores[proc_name],
                        'deductions': proc_deductions
                    })
    
    if not records:
        print("No evaluation scores found for cross-analysis")
        return
    
    # Phase 2: Aggregate using generic helper
    def aggregate_by(records, group_keys):
        """
        Group records by specified keys and calculate statistics.
        
        Args:
            records: List of record dictionaries
            group_keys: List of keys to group by (e.g., ['scenario', 'procedure'])
            
        Returns:
            Dictionary mapping group key tuples to aggregated data
        """
        from collections import defaultdict
        
        groups = defaultdict(lambda: {'scores': [], 'deductions': []})
        
        for record in records:
            # Create group key tuple
            key = tuple(record[k] for k in group_keys)
            groups[key]['scores'].append(record['score'])
            if record['deductions']:
                groups[key]['deductions'].append(record['deductions'])
        
        # Calculate statistics for each group
        result = {}
        for key, data in groups.items():
            scores = data['scores']
            deductions_list = data['deductions']
            
            # Calculate average deductions by category
            avg_deductions = {}
            if deductions_list:
                category_points = defaultdict(list)
                for deduction_dict in deductions_list:
                    for category, points in deduction_dict.items():
                        category_points[category].append(points)
                
                for category, points_list in category_points.items():
                    avg_deductions[category] = mean(points_list)
            
            result[key] = {
                'mean': mean(scores),
                'raw_scores': scores,
                'count': len(scores),
                'deductions': avg_deductions if avg_deductions else None
            }
        
        return result
    
    # Phase 3: Create output directories
    batch_dir = os.path.dirname(batch_experiment_dir)
    cross_analysis_dir = os.path.join(batch_dir, "cross_analysis")
    os.makedirs(cross_analysis_dir, exist_ok=True)
    
    # Phase 4: Generate charts
    
    # 4a. Scenario number charts (same scenario, different decoys)
    scenario_agg = aggregate_by(records, ['scenario', 'procedure'])
    
    # Group by scenario
    scenario_groups = defaultdict(dict)
    for (scenario, procedure), stats in scenario_agg.items():
        scenario_groups[scenario][procedure] = stats
    
    scenario_charts_dir = os.path.join(cross_analysis_dir, "by_scenario_number")
    os.makedirs(scenario_charts_dir, exist_ok=True)
    
    for scenario, proc_stats in scenario_groups.items():
        # Need at least 2 procedures or sufficient data
        if len(proc_stats) < 2:
            continue
        
        # Check if we have multiple takes (for meaningful comparison)
        total_evaluations = sum(stats['count'] for stats in proc_stats.values())
        if total_evaluations < 2:
            continue
        
        # Prepare data for chart
        avg_scores = {proc: proc_stats[proc]['mean'] for proc in procedure_labels if proc in proc_stats}
        raw_data = {proc: [[s] for s in proc_stats[proc]['raw_scores']] for proc in procedure_labels if proc in proc_stats}
        deduction_data = {proc: proc_stats[proc]['deductions'] for proc in procedure_labels if proc in proc_stats and proc_stats[proc]['deductions']}
        
        # Get take names for this scenario
        take_names = sorted(set(r['take'] for r in records if r['scenario'] == scenario))
        
        chart_path = os.path.join(scenario_charts_dir, f"scenario{scenario:02d}_cross_analysis_bar_chart.png")
        title = f"Scenario {scenario:02d} Cross-Analysis\n({', '.join(take_names)}, {total_evaluations} evaluations)"
        
        has_sufficient_data = any(len(proc_stats[proc]['raw_scores']) > 1 for proc in procedure_labels if proc in proc_stats)
        
        create_deduction_breakdown_bar_chart(
            avg_scores, procedure_labels, chart_path, title,
            raw_data if has_sufficient_data else None,
            deduction_data if deduction_data else None
        )
    
    # 4b. Method-wise decoy progression charts
    method_decoy_agg = aggregate_by(records, ['decoy', 'procedure'])
    
    # Reorganize by procedure then decoy
    method_decoy_groups = defaultdict(dict)
    for (decoy, procedure), stats in method_decoy_agg.items():
        method_decoy_groups[procedure][decoy] = stats
    
    method_charts_dir = os.path.join(cross_analysis_dir, "by_method_decoy_progression")
    os.makedirs(method_charts_dir, exist_ok=True)
    
    for proc_name in procedure_labels:
        if proc_name not in method_decoy_groups:
            continue
        
        decoy_stats = method_decoy_groups[proc_name]
        
        # Prepare data for progression chart
        decoy_data = {str(decoy): stats['raw_scores'] for decoy, stats in decoy_stats.items()}
        deduction_data = {str(decoy): stats['deductions'] for decoy, stats in decoy_stats.items() if stats['deductions']}
        
        chart_path = os.path.join(method_charts_dir, f"{proc_name.lower().replace(' ', '_')}_decoy_progression.png")
        
        create_method_decoy_progression_chart(
            decoy_data, proc_name, chart_path,
            deduction_data if deduction_data else None
        )
    
    # 4c. Overall comparison (all data)
    overall_agg = aggregate_by(records, ['procedure'])
    
    overall_charts_dir = os.path.join(cross_analysis_dir, "overall_comparison")
    os.makedirs(overall_charts_dir, exist_ok=True)
    
    if overall_agg:
        avg_scores = {proc: overall_agg[(proc,)]['mean'] for proc in procedure_labels if (proc,) in overall_agg}
        raw_data = {proc: [[s] for s in overall_agg[(proc,)]['raw_scores']] for proc in procedure_labels if (proc,) in overall_agg}
        deduction_data = {proc: overall_agg[(proc,)]['deductions'] for proc in procedure_labels if (proc,) in overall_agg and overall_agg[(proc,)]['deductions']}
        
        total_evaluations = sum(overall_agg[(proc,)]['count'] for proc in procedure_labels if (proc,) in overall_agg)
        
        chart_path = os.path.join(overall_charts_dir, "overall_methods_comparison.png")
        title = f"Overall Methods Comparison\n(All Scenarios & Decoys, {total_evaluations} evaluations)"
        
        has_sufficient_data = any(len(overall_agg[(proc,)]['raw_scores']) > 1 for proc in procedure_labels if (proc,) in overall_agg)
        
        create_deduction_breakdown_bar_chart(
            avg_scores, procedure_labels, chart_path, title,
            raw_data if has_sufficient_data else None,
            deduction_data if deduction_data else None
        )
    
    # Phase 5: Create summary
    unique_scenarios = sorted(set(r['scenario'] for r in records))
    unique_decoys = sorted(set(r['decoy'] for r in records))
    
    summary_data = {
        "split_name": split_name,
        "total_records": len(records),
        "scenario_numbers": unique_scenarios,
        "decoy_numbers": unique_decoys,
        "scenario_number_charts": len([s for s in scenario_groups if len(scenario_groups[s]) >= 2]),
        "overall_comparison_chart": 1 if overall_agg else 0,
        "method_progression_charts": len([p for p in method_decoy_groups if method_decoy_groups[p]]),
        "timestamp": datetime.now().isoformat()
    }
    
    summary_file = os.path.join(cross_analysis_dir, "cross_analysis_summary.json")
    with open(summary_file, 'w', encoding='utf-8') as f:
        json.dump(summary_data, f, indent=2, ensure_ascii=False)
    
    print(f"Cross-analysis charts created:")
    print(f"  - Scenario number charts: {scenario_charts_dir}")
    print(f"  - Overall methods comparison: {overall_charts_dir}")
    print(f"  - Method decoy progression charts: {method_charts_dir}")
    print(f"  - Summary: {summary_file}")


def create_batch_total_bar_chart(batch_experiment_dir):
    """
    Create aggregate bar chart for entire batch.
    
    Args:
        batch_experiment_dir: Directory containing all takes for this split (e.g., output/batch_XXXXXXXX/test)
    """
    procedure_labels = PROCEDURE_LABELS
    
    # Collect all aggregate results
    all_scores = []
    total_evaluations = 0
    take_count = 0
    
    for take_name in os.listdir(batch_experiment_dir):
        take_dir = os.path.join(batch_experiment_dir, take_name)
        if not os.path.isdir(take_dir):
            continue
        
        # Look for aggregate results
        aggregate_dir = os.path.join(take_dir, "aggregate")
        aggregate_results_file = os.path.join(aggregate_dir, "aggregate_results.json")
        
        if os.path.exists(aggregate_results_file):
            try:
                with open(aggregate_results_file, 'r', encoding='utf-8') as f:
                    aggregate_data = json.load(f)
                
                if "average_scores" in aggregate_data:
                    all_scores.append(aggregate_data["average_scores"])
                    total_evaluations += aggregate_data.get("total_iterations", 0)
                    take_count += 1
            except Exception as e:
                print(f"Error reading aggregate results for {take_name}: {e}")
                continue
    
    if not all_scores:
        print(f"No aggregate scores found for batch summary")
        return
    
    # Calculate overall average scores
    overall_avg_scores = {}
    for proc_name in procedure_labels:
        proc_scores = [s[proc_name] for s in all_scores if proc_name in s]
        if proc_scores:
            overall_avg_scores[proc_name] = mean(proc_scores)
        else:
            overall_avg_scores[proc_name] = 0
    
    # No detailed summary for batch level (removed as per user request)
    print(f"Batch summary: {take_count} takes, {total_evaluations} total evaluations")
