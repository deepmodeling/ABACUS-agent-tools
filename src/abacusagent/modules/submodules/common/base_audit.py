"""
Base audit logger for tracking parameter provenance.

This module provides the BaseAuditLogger class that all module-specific
audit loggers inherit from. It tracks the origin and reasoning for every
parameter value, enabling full traceability and reproducibility.
"""

import json
import uuid
from pathlib import Path
from typing import Dict, Any, List, Optional
from .base_schema import ParameterProvenance, AuditTrail, ValidationResult


class BaseAuditLogger:
    """
    Base audit logger for tracking parameter provenance.

    Provides common logging methods that all module-specific loggers inherit.
    Tracks the source, reasoning, and dependencies for every parameter value.

    Attributes:
        calculation_type: Type of calculation (e.g., "scf", "relax", "band")
        calculation_id: Unique identifier for this calculation
        provenances: Dictionary mapping parameter names to their provenance
        warnings: List of warning messages
        errors: List of error messages
        validation_results: List of validation results
    """

    def __init__(self, calculation_type: str, calculation_id: Optional[str] = None):
        """
        Initialize audit logger.

        Args:
            calculation_type: Type of calculation ("scf", "relax", "band", etc.)
            calculation_id: Optional unique ID for this calculation
                If not provided, a random 8-character ID is generated
        """
        self.calculation_type = calculation_type
        self.calculation_id = calculation_id or str(uuid.uuid4())[:8]
        self.provenances: Dict[str, ParameterProvenance] = {}
        self.warnings: List[str] = []
        self.errors: List[str] = []
        self.validation_results: List[ValidationResult] = []

    # ========================================================================
    # Provenance Logging Methods
    # ========================================================================

    def log_user_input(
        self,
        param_name: str,
        value: Any,
        reasoning: str = "Explicitly provided by user"
    ):
        """
        Log a parameter that was explicitly provided by the user.

        Args:
            param_name: Name of the parameter
            value: Value provided by user
            reasoning: Explanation (default: "Explicitly provided by user")
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
            value: Default value applied
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
            reasoning: Explanation of the inference logic
            depends_on: List of parameter names this value depends on
            inference_rule: Named identifier for the inference rule
                (e.g., "pulay_mixing_beta_default")
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
            reasoning: Explanation of the dependency constraint
            depends_on: List of parameter names this constraint depends on
        """
        self.provenances[param_name] = ParameterProvenance(
            parameter_name=param_name,
            value=value,
            source="dependency",
            reasoning=reasoning,
            depends_on=depends_on
        )

    # ========================================================================
    # Validation and Issue Tracking
    # ========================================================================

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

    def add_validation_result(self, result: ValidationResult):
        """
        Add a validation result to the audit trail.

        Args:
            result: ValidationResult object
        """
        self.validation_results.append(result)

    # ========================================================================
    # Audit Trail Generation
    # ========================================================================

    def create_audit_trail(self) -> AuditTrail:
        """
        Create the complete audit trail.

        Returns:
            AuditTrail object with all provenance and validation information
        """
        return AuditTrail(
            calculation_id=self.calculation_id,
            calculation_type=self.calculation_type,
            parameters=self.provenances,
            validation_results=self.validation_results,
            warnings=self.warnings,
            errors=self.errors
        )

    def save_audit_trail(self, output_path: Path) -> Path:
        """
        Save audit trail to JSON file.

        Args:
            output_path: Directory to save the audit trail file

        Returns:
            Path to the saved audit trail file
        """
        audit_trail = self.create_audit_trail()
        output_file = Path(output_path) / f"{self.calculation_type}_audit_{self.calculation_id}.json"

        with open(output_file, "w") as f:
            json.dump(audit_trail.to_dict(), f, indent=2)

        return output_file

    # ========================================================================
    # Human-Readable Output
    # ========================================================================

    def print_summary(self):
        """
        Print a human-readable summary of the audit trail to console.

        Displays:
        - Parameter provenance table with source and reasoning
        - Dependency information for inferred parameters
        - Warnings and errors
        - Validation summary
        """
        print(f"\n{'='*80}")
        print(f"{self.calculation_type.upper()} Calculation Audit Trail (ID: {self.calculation_id})")
        print(f"{'='*80}\n")

        # Print parameter provenance table
        if self.provenances:
            print("Parameter Provenance:")
            print(f"{'Parameter':<25} {'Value':<20} {'Source':<15} {'Reasoning'}")
            print(f"{'-'*80}")

            for param_name, prov in sorted(self.provenances.items()):
                # Format value for display
                value_str = self._format_value_for_display(prov.value)

                # Truncate long reasoning
                reasoning = prov.reasoning
                if len(reasoning) > 40:
                    reasoning = reasoning[:37] + "..."

                print(f"{param_name:<25} {value_str:<20} {prov.source:<15} {reasoning}")

                # Print dependency information if present
                if prov.depends_on:
                    deps = ", ".join(prov.depends_on)
                    print(f"{'':25} {'':20} {'':15}   └─ depends on: {deps}")

                # Print inference rule if present
                if prov.inference_rule:
                    print(f"{'':25} {'':20} {'':15}   └─ rule: {prov.inference_rule}")

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
            error_count = sum(1 for r in self.validation_results if r.severity == "error")
            warning_count = sum(1 for r in self.validation_results if r.severity == "warning")
            info_count = sum(1 for r in self.validation_results if r.severity == "info")

            print(f"\nValidation Summary:")
            print(f"  Errors: {error_count}, Warnings: {warning_count}, Info: {info_count}")

        print(f"\n{'='*80}\n")

    def get_summary_dict(self) -> Dict[str, Any]:
        """
        Get audit trail summary as a dictionary.

        Returns:
            Dictionary with summary statistics
        """
        return {
            "calculation_id": self.calculation_id,
            "calculation_type": self.calculation_type,
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

    # ========================================================================
    # Helper Methods
    # ========================================================================

    def _format_value_for_display(self, value: Any) -> str:
        """
        Format a parameter value for display in the summary table.

        Args:
            value: Parameter value to format

        Returns:
            Formatted string representation
        """
        if value is None:
            return "None"
        elif isinstance(value, float):
            # Use scientific notation for very small/large numbers
            if abs(value) < 0.01 or abs(value) > 1000:
                return f"{value:.2e}"
            else:
                return f"{value:.4f}"
        elif isinstance(value, bool):
            return str(value)
        elif hasattr(value, 'value'):  # Enum
            return str(value.value)
        else:
            value_str = str(value)
            # Truncate long values
            if len(value_str) > 18:
                return value_str[:15] + "..."
            return value_str
