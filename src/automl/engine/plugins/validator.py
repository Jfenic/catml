from __future__ import annotations

from typing import Any

from automl.domain.plugins.plugin import PluginCapability
from automl.domain.ports import PluginPort


class CompatibilityValidator:
    """Validates compatibility between plugins, tasks, and modalities

    to prevent expensive failures during trial execution.
    """

    @staticmethod
    def is_compatible(
        target: PluginPort | PluginCapability,
        task_type: str | None = None,
        modality: str | None = None,
    ) -> bool:
        caps = target.capabilities if hasattr(target, "capabilities") else target
        if task_type and not caps.is_compatible_with_task(task_type):
            return False
        if modality and not caps.is_compatible_with_modality(modality):
            return False
        return True

    @classmethod
    def validate_compatibility(
        cls,
        plugin_or_name: Any,
        capabilities: PluginCapability | None = None,
        task_type: str | None = None,
        modality: str | None = None,
    ) -> None:
        if isinstance(plugin_or_name, str):
            plugin_name = plugin_or_name
            caps = capabilities
        else:
            plugin_name = getattr(plugin_or_name, "plugin_id", str(plugin_or_name))
            caps = getattr(plugin_or_name, "capabilities", capabilities)

        if caps is None:
            raise ValueError(f"No capabilities provided for plugin '{plugin_name}'")

        if task_type and not caps.is_compatible_with_task(task_type):
            raise ValueError(
                f"Plugin '{plugin_name}' does not support task '{task_type}'. "
                f"Supported tasks: {caps.supported_tasks}"
            )
        if modality and not caps.is_compatible_with_modality(modality):
            raise ValueError(
                f"Plugin '{plugin_name}' does not support modality '{modality}'. "
                f"Supported modalities: {caps.supported_modalities}"
            )

    @staticmethod
    def validate_task_compatibility(plugin: PluginPort, task_type: str) -> None:
        if not plugin.capabilities.is_compatible_with_task(task_type):
            raise ValueError(
                f"Plugin '{plugin.plugin_id}' ({plugin.name}) is not compatible with task '{task_type}'. "
                f"Supported tasks: {plugin.capabilities.supported_tasks}"
            )

    @staticmethod
    def validate_modality_compatibility(plugin: PluginPort, modality: str) -> None:
        if not plugin.capabilities.is_compatible_with_modality(modality):
            raise ValueError(
                f"Plugin '{plugin.plugin_id}' ({plugin.name}) does not support modality '{modality}'. "
                f"Supported modalities: {plugin.capabilities.supported_modalities}"
            )
