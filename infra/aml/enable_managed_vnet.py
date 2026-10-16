"""One-off: enable the AML managed VNet (allow internet outbound) and recreate the GPU cluster
inside it with unchanged settings. Use it when the workspace storage account is private (e.g.
an organisation policy disables public network access), so compute must reach it through the
managed private endpoints. Managed network cannot be enabled while computes
exist, so the cluster is deleted first and recreated afterwards.

Usage: uv run --extra aml python infra/aml/enable_managed_vnet.py
"""

from __future__ import annotations

import json
import time

from azure.ai.ml.entities import AmlCompute, IdentityConfiguration, ManagedNetwork
from azure.core.exceptions import ResourceNotFoundError

from edb.runners.aml import ml_client
from edb.settings import get_settings

ALLOW_INTERNET_OUTBOUND = "AllowInternetOutbound"
# Fallback spec (the cluster as created originally) if it was already deleted.
DEFAULT_SPEC = {"size": "Standard_NC24ads_A100_v4", "tier": "low_priority", "min_instances": 0,
                "max_instances": 12, "idle_time_before_scale_down": 120}


def main() -> None:
    s = get_settings()
    s.require("aml_workspace", "aml_compute")
    ml = ml_client()

    try:
        d = ml.compute.get(s.aml_compute)._to_dict()
        spec = {k: d.get(k, v) for k, v in DEFAULT_SPEC.items()}
        print(f"deleting cluster {s.aml_compute}: {json.dumps(spec)}")
        ml.compute.begin_delete(s.aml_compute).result()
    except ResourceNotFoundError:
        spec = DEFAULT_SPEC

    ws = ml.workspaces.get(s.aml_workspace)
    mode = getattr(ws.managed_network, "isolation_mode", None)
    print(f"workspace isolation mode: {mode}")
    if ALLOW_INTERNET_OUTBOUND not in str(getattr(mode, "value", mode)):
        ws.managed_network = ManagedNetwork(isolation_mode=ALLOW_INTERNET_OUTBOUND)
        ml.workspaces.begin_update(ws).result()
        print("isolation mode updated")
    t0 = time.time()
    ml.workspaces.begin_provision_network(workspace_name=s.aml_workspace,
                                          include_spark=False).result()
    print(f"managed network provisioned in {time.time() - t0:.0f}s")

    ml.compute.begin_create_or_update(AmlCompute(
        name=s.aml_compute, identity=IdentityConfiguration(type="system_assigned"), **spec,
    )).result()
    c = ml.compute.get(s.aml_compute)
    print("cluster identity principal:", c.identity.principal_id,
          "-> grant it 'Storage Blob Data Contributor' on the workspace storage account")
    print("cluster:", json.dumps({k: str(v) for k, v in c._to_dict().items()}))


if __name__ == "__main__":
    main()
