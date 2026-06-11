import yaml
from dataclasses import dataclass


DEFAULT_EXTS = [".js", ".jsx", ".ts", ".tsx"]

@dataclass
class ServiceConfig:
    name: str
    system_prompt: str
    repos: list[str]
    file_extensions: list[str] = None
    jira_project_key: str = None

    def __post_init__(self):
        if self.file_extensions is None:
            self.file_extensions = DEFAULT_EXTS


def load_services(path: str) -> list[ServiceConfig]:
    with open(path) as f:
        data = yaml.safe_load(f)
    return [
        ServiceConfig(
            name=svc["name"],
            system_prompt=svc["system_prompt"],
            repos=svc["repos"],
            file_extensions=svc.get("file_extensions"),
            jira_project_key=svc.get("jira_project_key"),
        )
        for svc in data["services"]
    ]
