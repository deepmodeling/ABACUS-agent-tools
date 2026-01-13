"""
Audit trail and provenance tracking for SCF parameters.

This module implements the traceability principle:
- Every parameter value has documented origin
- Full audit trail from user input → defaults → inference → final value
- Human-readable summaries and machine-readable JSON output
"""

import json
import uuid
from pathlib import Path
from typing import Dict, Any, List, Optional

from .schema import ParameterProvenance, SCFAuditTrail


class SCFAuditLogger:
    """
    Tracks parameter provenance and creates audit trails.

    Design principle: Every parameter value must have a documented origin.
    This class provides methods to log different types of parameter sources
    and generate comprehensive audit trails.

    Usage:
        audit = SCFAuditLogger()
        audit.log_user_input("ecutwfc", 100, "Explicitly provided by user")
        audit.log_default("scf_thr", 1e-6, "Standard convergence threshold")
        trail = audit.create_audit_trail()
        audit.print_summary()
    """

    def __init__(self, calculation_id: Optional[str] = None):
        """
        Initialize audit logger.

        Args:
            calculation_id: Optional unique ID for this calculation.
                           If not provided, generates a random 8-character ID.
        """
        self.calculation_id = calculation_id or str(uuid.uuid4())[:8]
        self.provenances: Dict[str, ParameterProvenance] = {}
        self.warnings: List[str] = []
        self.errors: List[str] = []
        self.validation_results: List[dict] = []

    def log_user_input(
        self,
        param_name: str,
        value: Any,
        reasoning: str = "Explicitly provided by user"
    ):
        """
        Log a parameter that was explicitly provided by the user.

        Args:
            param_name: Name of the parameter (e.g., 'ecutwfc')
            value: Value provided by user
            reasoning: Explanation of the value (default: standard message)
        """
        self.provenances[param_name] = ParameterProvenance(
            parameter_name=param_name,
            value=value,
            source="user_input",
            reasoning=reasoning
        )

    def log_default(self, param_name: str, value: Any, reasoning: str):
        """
        Log a parameter that uses a default value.

        Args:
            param_name: Name of the parameter
            value: Default value
            reasoning: Explanation of why this default was chosen
        """
        self.provenances[param_name] = ParameterProvenance(
            parameter_name=param_name,
            value=value,
            source="default",
            reasoning=reasoning
        )

    def log_inferred(
        self,
        param_name: str,
        value: Any,
        reasoning: str,
        depends_on: List[str],
        inference_rule: str
    ):
        """
        Log a parameter that was inferred from other parameters.

        Args:
            param_name: Name of the parameter
            value: Inferred value
            reasoning: Explanation of the inference
            depends_on: List of parameter names this depends on
            inference_rule: Name of the inference rule applied
        """
        self.provenances[param_name] = ParameterProvenance(
            parameter_name=param_name,
            value=value,
            source="inferred",
            reasoning=reasoning,
            depends_on=depends_on,
            inference_rule=inference_rule
        )

    def log_dependency(
        self,
        param_name: str,
        value: Any,
        reasoning: str,
        depends_on: List[str]
    ):
        """
        Log a parameter that was set due to dependency constraints.

        Args:
            param_name: Name of the parameter
            value: Value set by dependency
            reasoning: Explanation of the dependency
            depends_on: List of parameter names this depends on
        """
        self.provenances[param_name] = ParameterProvenance(
            parameter_name=param_name,
            value=value,
            source="dependency",
            reasoning=reasoning,
            depends_on=depends_on
        )

    def add_warning(self, message: str):
        """
        Add a warning message to the audit trail.

        Args:
            message: Warning message
        """
        self.warnings.append(message)

    def add_error(self, message: str):
        """
        Add an error message to the audit trail.

        Args:
            message: Error message
        """
        self.errors.append(message)

    def add_validation_result(self, result: dict):
        """
        Add a validation result to the audit trail.

        Args:
            result: Validation result dictionary
        """
        self.validation_results.append(result)

    def create_audit_trail(self) -> SCFAuditTrail:
        """
        Create the complete audit trail.

        Returns:
            SCFAuditTrail object containing all provenance information
        """
        return SCFAuditTrail(
            calculation_id=self.calculation_id,
            parameters=self.provenances,
            validation_results=self.validation_results,
            warnings=self.warnings,
            errors=self.errors
        )

    def save_audit_trail(self, output_path: Path):
        """
        Save audit trail to JSON file.

        Args:
            output_path: Directory to save the audit trail file
        """
        audit_trail = self.create_audit_trail()
        output_file = Path(output_path) / f"scf_audit_{self.calculation_id}.json"

        with open(output_file, "w") as f:
            json.dump(audit_trail.to_dict(), f, indent=2)

        return output_file

    def print_summary(self):
        """
        Print a human-readable summary of the audit trail to console.

        This provides a clear overview of:
        - Parameter provenance (where each value came from)
        - Warnings (non-critical issues)
        - Errors (critical issues that block execution)
        """
        print(f"\n{'='*80}")
        print(f"SCF Calculation Audit Trail (ID: {self.calculation_id})")
        print(f"{'='*80}\n")

        # Print parameter provenance table
        print("Parameter Provenance:")
        print(f"{'Parameter':<25} {'Value':<20} {'Source':<15} {'Reasoning'}")
        print(f"{'-'*80}")

        for param_name, prov in sorted(self.provenances.items()):
            # Format value for display
            if prov.value is None:
                value_str = "None"
            elif isinstance(prov.value, float):
                # Use scientific notation for very small/large numbers
                if abs(prov.value) < 0.01 or abs(prov.value) > 1000:
                    value_str = f"{prov.value:.2e}"
                else:
                    value_str = f"{prov.value:.4f}"
            elif isinstance(prov.value, bool):
                value_str = str(prov.value)
            else:
                value_str = str(prov.value)

            # Truncate long values
            if len(value_str) > 18:
                value_str = value_str[:15] + "..."

            # Truncate long reasoning
            reasoning = prov.reasoning
            if len(reasoning) > 40:
                reasoning = reasoning[:37] + "..."

            print(f"{param_name:<25} {value_str:<20} {prov.source:<15} {reasoning}")

            # Print dependency information if present
            if prov.depends_on:
                deps = ", ".join(prov.depends_on)
                print(f"{'':25} {'':20} {'':15}   └─ depends on: {deps}")

        # Print warnings
        if self.warnings:
            print(f"\n⚠ Warnings ({len(self.warnings)}):")
            for warning in self.warnings:
                print(f"  • {warning}")

        # Print errors
        if self.errors:
            print(f"\n✗ Errors ({len(self.errors)}):")
            for error in self.errors:
                print(f"  • {error}")

        # Print validation summary
        if self.validation_results:
            error_count = sum(1 for r in self.validation_results if r.get("severity") == "error")
            warning_count = sum(1 for r in self.validation_results if r.get("severity") == "warning")
            info_count = sum(1 for r in self.validation_results if r.get("severity") == "info")

            print(f"\nValidation Summary:")
            print(f"  Errors: {error_count}, Warnings: {warning_count}, Info: {info_count}")

        print(f"\n{'='*80}\n")

    def get_summary_dict(self) -> dict:
        """
        Get audit trail summary as a dictionary.

        Returns:
            Dictionary containing summary information suitable for
            including in calculation results.
        """
        return {
            "calculation_id": self.calculation_id,
            "parameter_count": len(self.provenances),
            "sources": {
                "user_input": sum(1 for p in self.provenances.values() if p.source == "user_input"),
                "default": sum(1 for p in self.provenances.values() if p.source == "default"),
                "inferred": sum(1 for p in self.provenances.values() if p.source == "inferred"),
                "dependency": sum(1 for p in self.provenances.values() if p.source == "dependency"),
            },
            "warnings_count": len(self.warnings),
            "errors_count": len(self.errors),
            "has_errors": len(self.errors) > 0,
        }
