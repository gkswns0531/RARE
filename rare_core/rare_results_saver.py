import os
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, Any
from dataclasses import asdict

from rare_entities import (
    RedundancyPipelineResult,
    AtomicInfo,
    RedundancyMapping,
    RareEvaluationItem,
)


class ResultsSaver:
    """Save RARE pipeline results to files"""
    
    def __init__(self, output_dir: str = "rare_results"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
    def save_complete_results(self, result: RedundancyPipelineResult, prefix: str = "pipeline") -> Dict[str, str]:
        """Save all pipeline results to separate files"""
        
        saved_files = {}
        
        # 1. Save atomic information mapping
        atomic_file = self.output_dir / f"{prefix}_atomic_info_{self.timestamp}.json"
        self._save_atomic_info_mapping(result.atomic_info_map, atomic_file)
        saved_files["atomic_info"] = str(atomic_file)
        
        # 2. Save redundancy mapping
        redundancy_file = self.output_dir / f"{prefix}_redundancy_mapping_{self.timestamp}.json"
        self._save_redundancy_mapping(result.redundancy_mapping, redundancy_file)
        saved_files["redundancy_mapping"] = str(redundancy_file)
        
        # 3. Save evaluation items
        evaluation_file = self.output_dir / f"{prefix}_evaluation_items_{self.timestamp}.json"
        self._save_evaluation_items(result.evaluation_items, evaluation_file)
        saved_files["evaluation_items"] = str(evaluation_file)
        
        # 4. Save summary report
        summary_file = self.output_dir / f"{prefix}_summary_{self.timestamp}.json"
        self._save_summary_report(result, summary_file)
        saved_files["summary"] = str(summary_file)
        
        # 5. Save readable text report
        report_file = self.output_dir / f"{prefix}_report_{self.timestamp}.txt"
        self._save_readable_report(result, report_file)
        saved_files["report"] = str(report_file)
        
        return saved_files
    
    def _save_atomic_info_mapping(self, atomic_info_map: Dict[str, Any], file_path: Path):
        """Save atomic information mapping to JSON"""
        
        # Convert AtomicInfo objects to dict
        json_data = {
            "timestamp": datetime.now().isoformat(),
            "total_chunks": len(atomic_info_map),
            "total_atomic_info": sum(len(info_list) for info_list in atomic_info_map.values()),
            "atomic_info_map": {}
        }
        
        for chunk_id, atomic_info_list in atomic_info_map.items():
            json_data["atomic_info_map"][chunk_id] = [
                asdict(atomic_info) for atomic_info in atomic_info_list
            ]
        
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(json_data, f, ensure_ascii=False, indent=2)
            
        print(f"Atomic info mapping saved: {file_path}")
    
    def _save_redundancy_mapping(self, redundancy_mapping: Dict[str, RedundancyMapping], file_path: Path):
        """Save redundancy mapping to JSON"""
        
        unique_count = sum(1 for r in redundancy_mapping.values() if r.redundant_items == ["unique"])
        redundant_count = len(redundancy_mapping) - unique_count
        
        json_data = {
            "timestamp": datetime.now().isoformat(),
            "total_atomic_info": len(redundancy_mapping),
            "unique_count": unique_count,
            "redundant_count": redundant_count,
            "redundancy_rate": round(redundant_count / len(redundancy_mapping) * 100, 1) if redundancy_mapping else 0,
            "redundancy_mapping": {}
        }
        
        for atomic_id, mapping in redundancy_mapping.items():
            json_data["redundancy_mapping"][atomic_id] = asdict(mapping)
        
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(json_data, f, ensure_ascii=False, indent=2)
            
        print(f"Redundancy mapping saved: {file_path}")
    
    def _save_evaluation_items(self, evaluation_items: list[RareEvaluationItem], file_path: Path):
        """Save evaluation items to JSON"""
        
        json_data = {
            "timestamp": datetime.now().isoformat(),
            "total_items": len(evaluation_items),
            "evaluation_items": [asdict(item) for item in evaluation_items]
        }
        
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(json_data, f, ensure_ascii=False, indent=2)
            
        print(f"Evaluation items saved: {file_path}")
    
    def _save_summary_report(self, result: RedundancyPipelineResult, file_path: Path):
        """Save summary statistics and cost information"""
        
        json_data = {
            "timestamp": datetime.now().isoformat(),
            "pipeline_statistics": result.statistics,
            "cost_summary": asdict(result.cost_summary),
            "redundancy_summary": {
                "total_atomic_info": len(result.redundancy_mapping),
                "unique_items": sum(1 for r in result.redundancy_mapping.values() if r.redundant_items == ["unique"]),
                "redundant_items": sum(1 for r in result.redundancy_mapping.values() if r.redundant_items != ["unique"]),
            },
            "evaluation_summary": {
                "total_generated": len(result.evaluation_items),
                "by_redundancy_level": self._get_redundancy_level_stats(result.evaluation_items),
                "by_importance_score": self._get_importance_score_stats(result.evaluation_items),
            }
        }
        
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(json_data, f, ensure_ascii=False, indent=2)
            
        print(f"Summary report saved: {file_path}")
    
    def _save_readable_report(self, result: RedundancyPipelineResult, file_path: Path):
        """Save human-readable text report"""
        
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write("RARE Pipeline Results Report\n")
            f.write("=" * 50 + "\n")
            f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            # Statistics
            stats = result.statistics
            f.write("Pipeline Statistics:\n")
            f.write(f"  - Documents processed: {stats.get('total_documents', 0)}\n")
            f.write(f"  - Total atomic info: {stats.get('total_atomic_info', 0)}\n")
            f.write(f"  - Unique atomic info: {stats.get('unique_atomic_info', 0)}\n")
            f.write(f"  - Redundant atomic info: {stats.get('redundant_atomic_info', 0)}\n")
            f.write(f"  - Generated evaluation items: {stats.get('generated_evaluation_items', 0)}\n")
            f.write(f"  - Total API calls: {stats.get('total_api_calls', 0)}\n\n")
            
            # Cost information
            cost = result.cost_summary
            f.write("Cost Summary:\n")
            f.write(f"  - Total cost: ${cost.total_cost_usd:.6f} USD\n")
            f.write(f"  - Total tokens: {cost.total_tokens:,}\n")
            f.write(f"  - Prompt tokens: {cost.prompt_tokens:,}\n")
            f.write(f"  - Completion tokens: {cost.completion_tokens:,}\n")
            f.write(f"  - API calls: {cost.total_calls}\n\n")
            
            # Sample evaluation items
            f.write("Sample Evaluation Items:\n")
            f.write("-" * 40 + "\n")
            for i, item in enumerate(result.evaluation_items[:5], 1):  # Show first 5
                f.write(f"\n{i}. Redundancy Level: {item.redundancy_level}\n")
                f.write(f"   Importance Score: {item.importance_score}\n")
                f.write(f"   Question: {item.question}\n")
                f.write(f"   Answer: {item.target_answer}\n")
                f.write(f"   Atomic Info: {item.atomic_info[:100]}...\n")
                
        print(f"Readable report saved: {file_path}")
    
    def _get_redundancy_level_stats(self, evaluation_items: list[RareEvaluationItem]) -> Dict[str, int]:
        """Get statistics by redundancy level"""
        stats = {}
        for item in evaluation_items:
            level = str(item.redundancy_level)
            stats[level] = stats.get(level, 0) + 1
        return stats
    
    def _get_importance_score_stats(self, evaluation_items: list[RareEvaluationItem]) -> Dict[str, int]:
        """Get statistics by importance score"""
        stats = {}
        for item in evaluation_items:
            score = str(item.importance_score)
            stats[score] = stats.get(score, 0) + 1
        return stats
