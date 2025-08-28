from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List

from k3dgen.house import House


class Condo:
    """A collection of K3D Houses, acting as a router for different experts."""

    def __init__(self, config_path: str | Path):
        """Initialize the Condo with a configuration file.

        Parameters
        ----------
        config_path:
            Path to the condo.json configuration file.
        """
        self.config_path = Path(config_path)
        if not self.config_path.exists():
            raise FileNotFoundError(f"Condo configuration not found at {self.config_path}")

        self.config = json.loads(self.config_path.read_text())
        self._validate_config()

        self._house_cache: Dict[str, House] = {}

    def _validate_config(self):
        """Validate the structure of the condo.json file."""
        if "houses" not in self.config or not isinstance(self.config["houses"], list):
            raise ValueError("Condo config must have a 'houses' list.")
        for house_info in self.config["houses"]:
            if not all(k in house_info for k in ["uri", "expert"]):
                raise ValueError("Each house entry must have a 'uri' and 'expert' key.")

    def list_experts(self) -> List[str]:
        """Return a list of all expert names available in the condo."""
        return [house_info["expert"] for house_info in self.config["houses"]]

    def get_house(self, expert_name: str) -> House:
        """Get a House instance for a specific expert.

        Houses are cached after their first retrieval.

        Parameters
        ----------
        expert_name:
            The name of the expert whose house is to be retrieved.

        Returns
        -------
        A House instance.
        """
        if expert_name in self._house_cache:
            return self._house_cache[expert_name]

        for house_info in self.config["houses"]:
            if house_info["expert"] == expert_name:
                uri = house_info["uri"]

                # Resolve relative file paths
                if uri.startswith("file://./"):
                    base_dir = self.config_path.parent
                    relative_path = uri.removeprefix("file://./")
                    absolute_path = base_dir.joinpath(relative_path).resolve()
                    uri = absolute_path.as_uri()

                house = House(uri)
                self._house_cache[expert_name] = house
                return house

        raise KeyError(f"Expert '{expert_name}' not found in condo configuration.")
