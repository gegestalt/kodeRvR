"""Central project paths and non-destructive dataset discovery."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class ProjectPaths:
    root: Path

    @classmethod
    def discover(cls, start: Path | None = None) -> "ProjectPaths":
        current = (start or Path.cwd()).resolve()
        for candidate in (current, *current.parents):
            if (candidate / "src" / "ips").is_dir() and (candidate / "README.md").exists():
                return cls(candidate)
        raise FileNotFoundError("could not locate project root containing src/ips and README.md")

    @property
    def data(self) -> Path:
        return self.root / "data"

    @property
    def results(self) -> Path:
        return self.root / "results" / "notebook_ips_lab"

    @property
    def cicapt_canonical(self) -> Path:
        return self.data / "cicapt_iiot2024" / "raw"

    @property
    def cicapt_browser_download(self) -> Path:
        return self.root / "CICAPT-IIoT2024"

    def cicapt_source(self) -> Path | None:
        for candidate in (self.cicapt_canonical, self.cicapt_browser_download):
            if candidate.is_dir() and any(candidate.iterdir()):
                return candidate
        return None

    def cicapt_inventory(self) -> dict[str, list[Path]]:
        source = self.cicapt_source()
        output = {"network_csv": [], "pcap": [], "provenance": [], "attack_info": [], "other": []}
        if source is None:
            return output
        for path in sorted(item for item in source.rglob("*") if item.is_file()):
            lower = path.name.casefold()
            if "attack_info" in lower or "attack info" in lower:
                output["attack_info"].append(path)
            elif "provenance" in lower or "spade" in lower or "graph" in lower:
                output["provenance"].append(path)
            elif path.suffix.casefold() in {".pcap", ".pcapng"}:
                output["pcap"].append(path)
            elif path.suffix.casefold() == ".csv" and "network" in lower:
                output["network_csv"].append(path)
            else:
                output["other"].append(path)
        return output

    def cicapt_status(self) -> dict[str, object]:
        source = self.cicapt_source()
        inventory = self.cicapt_inventory()
        required = ("network_csv", "provenance", "attack_info")
        missing = [name for name in required if not inventory[name]]
        return {
            "download_detected": source is not None,
            "source": str(source) if source else None,
            "canonical_location": str(self.cicapt_canonical),
            "ready_for_audit": not missing,
            "missing_modalities": missing,
            "counts": {name: len(files) for name, files in inventory.items()},
            "bytes": sum(path.stat().st_size for files in inventory.values() for path in files),
        }

    def cicapt_primary_artifacts(self) -> dict[str, Path]:
        """Return the one authoritative file for each CICAPT experiment role."""
        source = self.cicapt_source()
        if source is None:
            raise FileNotFoundError("CICAPT download not found")
        canonical = {
            "phase1_network": source / "network" / "phase1_NetworkData.csv",
            "phase2_network": source / "network" / "phase2_NetworkData.csv",
            "phase1_provenance": source / "provenance" / "Phase1_Provenance.csv",
            "phase2_provenance": source / "provenance" / "Phase2_Provenance.csv",
            "attack_info": source / "ground_truth" / "attack_info.csv",
        }
        missing = [str(path) for path in canonical.values() if not path.exists()]
        if missing:
            raise FileNotFoundError(f"canonical CICAPT artifacts missing: {missing}")
        return canonical
