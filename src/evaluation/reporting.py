"""
Reporting module for evaluation framework.
Supports JSON, CSV, and Markdown output formats.
"""
import json
import csv
import logging
from pathlib import Path
from typing import Dict, Any, Optional
from datetime import datetime

from .models import EvaluationReport, ExperimentComparison

logger = logging.getLogger(__name__)


class ReportGenerator:
    """
    Generates evaluation reports in multiple formats.
    
    Supported formats:
    - JSON: Full detailed report
    - CSV: Per-query results and aggregate metrics
    - Markdown: Human-readable summary with GitHub-friendly formatting
    """
    
    @staticmethod
    def generate_json(report: EvaluationReport, output_path: str) -> None:
        """
        Generate JSON report.
        
        Args:
            report: Evaluation report
            output_path: Path to save JSON file
        """
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(report.to_dict(), f, indent=2, default=str)
        
        logger.info(f"JSON report saved to: {output_file}")
    
    @staticmethod
    def generate_csv(report: EvaluationReport, output_dir: str) -> None:
        """
        Generate CSV reports (separate files for different data).
        
        Args:
            report: Evaluation report
            output_dir: Directory to save CSV files
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Aggregate metrics CSV
        ReportGenerator._write_aggregate_metrics_csv(report, output_path)
        
        # Per-query results CSV
        ReportGenerator._write_per_query_csv(report, output_path)
        
        # Latency metrics CSV
        ReportGenerator._write_latency_metrics_csv(report, output_path)
        
        logger.info(f"CSV reports saved to: {output_path}")
    
    @staticmethod
    def _write_aggregate_metrics_csv(report: EvaluationReport, output_path: Path) -> None:
        """Write aggregate metrics to CSV."""
        filepath = output_path / "aggregate_metrics.csv"
        
        with open(filepath, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(["metric", "value"])
            
            for metric_name, value in report.aggregate_metrics.items():
                writer.writerow([metric_name, value])
    
    @staticmethod
    def _write_per_query_csv(report: EvaluationReport, output_path: Path) -> None:
        """Write per-query results to CSV."""
        filepath = output_path / "per_query_results.csv"
        
        with open(filepath, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            
            # Header
            header = ["query_id", "query_text"]
            if report.query_results:
                metric_names = list(report.query_results[0].metrics.keys())
                header.extend(metric_names)
            header.extend(["latency", "chunks_returned", "avg_similarity", "zero_recall"])
            writer.writerow(header)
            
            # Data rows
            for qr in report.query_results:
                row = [qr.query.query_id, qr.query.query_text]
                for metric_name in metric_names:
                    row.append(qr.metrics.get(metric_name, 0.0))
                row.extend([
                    qr.latency,
                    qr.chunks_returned,
                    qr.avg_similarity,
                    qr.get_zero_recall()
                ])
                writer.writerow(row)
    
    @staticmethod
    def _write_latency_metrics_csv(report: EvaluationReport, output_path: Path) -> None:
        """Write latency metrics to CSV."""
        filepath = output_path / "latency_metrics.csv"
        
        with open(filepath, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(["metric", "value"])
            
            for metric_name, value in report.latency_metrics.items():
                writer.writerow([metric_name, value])
    
    @staticmethod
    def generate_markdown(report: EvaluationReport, output_path: str) -> None:
        """
        Generate Markdown report (GitHub-friendly).
        
        Args:
            report: Evaluation report
            output_path: Path to save Markdown file
        """
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        markdown = ReportGenerator._build_markdown_report(report)
        
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(markdown)
        
        logger.info(f"Markdown report saved to: {output_file}")
    
    @staticmethod
    def _build_markdown_report(report: EvaluationReport) -> str:
        """Build markdown report string."""
        lines = []
        
        # Title
        lines.append("# Evaluation Report")
        lines.append("")
        
        # Experiment Metadata
        lines.append("## Experiment Metadata")
        lines.append("")
        lines.append(f"- **Experiment ID**: `{report.metadata.experiment_id}`")
        lines.append(f"- **Timestamp**: {report.metadata.timestamp.strftime('%Y-%m-%d %H:%M:%S')}")
        lines.append(f"- **Dataset**: {report.metadata.dataset}")
        lines.append(f"- **Retriever**: {report.metadata.retriever}")
        lines.append(f"- **Embedding Model**: {report.metadata.embedding_model}")
        lines.append(f"- **Vector Database**: {report.metadata.vector_database}")
        lines.append(f"- **Collection**: {report.metadata.collection}")
        lines.append("")
        
        # Aggregate Metrics
        lines.append("## Aggregate Metrics")
        lines.append("")
        lines.append("| Metric | Value |")
        lines.append("|--------|-------|")
        
        # Filter out standard deviation metrics for cleaner table
        for metric_name, value in report.aggregate_metrics.items():
            if not metric_name.endswith("_std"):
                lines.append(f"| {metric_name} | {value:.4f} |")
        
        lines.append("")
        
        # Latency Metrics
        lines.append("## Latency Metrics")
        lines.append("")
        lines.append("| Metric | Value (s) |")
        lines.append("|--------|-----------|")
        
        for metric_name, value in report.latency_metrics.items():
            lines.append(f"| {metric_name} | {value:.4f} |")
        
        lines.append("")
        
        # Failure Analysis
        lines.append("## Failure Analysis")
        lines.append("")
        lines.append(f"- **Zero Recall Queries**: {report.failure_analysis['zero_recall_count']}/{report.failure_analysis['total_queries']}")
        
        if report.failure_analysis['zero_recall_queries']:
            lines.append("- **Zero Recall Query IDs**:")
            for qid in report.failure_analysis['zero_recall_queries']:
                lines.append(f"  - `{qid}`")
        
        lines.append("")
        lines.append("### Most Frequently Missed Chunks")
        lines.append("")
        lines.append("| Chunk ID | Miss Count |")
        lines.append("|----------|------------|")
        
        for chunk_id, count in report.failure_analysis['most_missed_chunks']:
            lines.append(f"| `{chunk_id}` | {count} |")
        
        lines.append("")
        
        # Per-Query Results
        lines.append("## Per-Query Results")
        lines.append("")
        
        for qr in report.query_results:
            lines.append(f"### Query: {qr.query.query_id}")
            lines.append("")
            lines.append(f"**Text**: {qr.query.query_text}")
            lines.append("")
            lines.append("**Metrics**:")
            lines.append("")
            for metric_name, value in qr.metrics.items():
                lines.append(f"- {metric_name}: {value:.4f}")
            lines.append("")
            lines.append(f"- Latency: {qr.latency:.4f}s")
            lines.append(f"- Chunks Returned: {qr.chunks_returned}")
            lines.append(f"- Avg Similarity: {qr.avg_similarity:.4f}")
            lines.append("")
            
            if qr.get_missing_chunks():
                lines.append("**Missing Chunks**:")
                for chunk_id in qr.get_missing_chunks():
                    lines.append(f"- `{chunk_id}`")
                lines.append("")
            
            lines.append("---")
            lines.append("")
        
        return "\n".join(lines)
    
    @staticmethod
    def generate_comparison_markdown(comparison: ExperimentComparison, output_path: str) -> None:
        """
        Generate Markdown comparison report.
        
        Args:
            comparison: Experiment comparison
            output_path: Path to save Markdown file
        """
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        markdown = ReportGenerator._build_comparison_markdown(comparison)
        
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(markdown)
        
        logger.info(f"Comparison markdown saved to: {output_file}")
    
    @staticmethod
    def _build_comparison_markdown(comparison: ExperimentComparison) -> str:
        """Build comparison markdown string."""
        lines = []
        
        # Title
        lines.append("# Experiment Comparison")
        lines.append("")
        
        # Experiment Info
        lines.append("## Experiments")
        lines.append("")
        lines.append(f"- **Experiment A**: `{comparison.experiment_a.metadata.experiment_id}`")
        lines.append(f"  - Timestamp: {comparison.experiment_a.metadata.timestamp.strftime('%Y-%m-%d %H:%M:%S')}")
        lines.append(f"  - Dataset: {comparison.experiment_a.metadata.dataset}")
        lines.append(f"  - Retriever: {comparison.experiment_a.metadata.retriever}")
        lines.append("")
        lines.append(f"- **Experiment B**: `{comparison.experiment_b.metadata.experiment_id}`")
        lines.append(f"  - Timestamp: {comparison.experiment_b.metadata.timestamp.strftime('%Y-%m-%d %H:%M:%S')}")
        lines.append(f"  - Dataset: {comparison.experiment_b.metadata.dataset}")
        lines.append(f"  - Retriever: {comparison.experiment_b.metadata.retriever}")
        lines.append("")
        
        # Metric Comparison Table
        lines.append("## Metric Comparison")
        lines.append("")
        lines.append("| Metric | Experiment A | Experiment B | Difference | Change |")
        lines.append("|--------|--------------|--------------|------------|--------|")
        
        for metric_name, diff_data in comparison.metric_differences.items():
            if not metric_name.endswith("_std"):
                value_a = diff_data["value_a"]
                value_b = diff_data["value_b"]
                difference = diff_data["difference"]
                percent_change = diff_data["percent_change"]
                
                # Format change with emoji
                if percent_change > 0:
                    change_emoji = "📈"
                    change_str = f"+{percent_change:.2f}%"
                elif percent_change < 0:
                    change_emoji = "📉"
                    change_str = f"{percent_change:.2f}%"
                else:
                    change_emoji = "➡️"
                    change_str = "0.00%"
                
                lines.append(
                    f"| {metric_name} | {value_a:.4f} | {value_b:.4f} | "
                    f"{difference:+.4f} | {change_emoji} {change_str} |"
                )
        
        lines.append("")
        
        # Improvement Summary
        lines.append("## Improvement Summary")
        lines.append("")
        
        if comparison.improvement_summary["improved"]:
            lines.append("### ✅ Improved Metrics")
            lines.append("")
            for item in comparison.improvement_summary["improved"]:
                lines.append(f"- **{item['metric']}**: +{item['percent_change']:.2f}%")
            lines.append("")
        
        if comparison.improvement_summary["degraded"]:
            lines.append("### ❌ Degraded Metrics")
            lines.append("")
            for item in comparison.improvement_summary["degraded"]:
                lines.append(f"- **{item['metric']}**: {item['percent_change']:.2f}%")
            lines.append("")
        
        if comparison.improvement_summary["unchanged"]:
            lines.append("### ➡️ Unchanged Metrics")
            lines.append("")
            for metric in comparison.improvement_summary["unchanged"]:
                lines.append(f"- {metric}")
            lines.append("")
        
        return "\n".join(lines)
