<#
.SYNOPSIS
  Provision the Azure ML workspace + GPU compute instance for open-weight model runs.

.DESCRIPTION
  1. Checks Azure ML dedicated vCPU quota for the requested GPU size (fails fast if 0).
  2. Creates the workspace if it does not exist.
  3. Creates the compute instance (system-assigned identity, idle shutdown, setup script).
  4. Grants the compute identity "Cognitive Services User" on the Foundry resource so the
     benchmark can call gpt-6-luna / Decision-1 from the VM with Entra ID.

.EXAMPLE
  .\infra\aml\provision.ps1 -CheckOnly
  .\infra\aml\provision.ps1 -Size Standard_NC24ads_A100_v4
  .\infra\aml\provision.ps1 -Location eastus2 -Workspace kz-exp-aml-eus2 -Size Standard_NC40ads_H100_v5
#>
[CmdletBinding()]
param(
    [string]$ResourceGroup = "kz-exp-rg",
    [string]$Workspace = "kz-exp-aml",
    [string]$Location = "swedencentral",
    [string]$ComputeName = "kz-gpu-ci",
    [string]$Size = "Standard_NC24ads_A100_v4",
    [string]$FoundryResource = "llm-decisions-resource",
    [switch]$CheckOnly
)
$ErrorActionPreference = "Stop"
$here = $PSScriptRoot

# GPU sizes suitable for gpt-oss-20b -> Azure ML quota family name + vCPUs.
$sizes = @{
    "Standard_NV36ads_A10_v5"   = @{ Family = "StandardNVADSA10v5Family";     VCpus = 36 }
    "Standard_NC24ads_A100_v4"  = @{ Family = "standardNCADSA100v4Family";    VCpus = 24 }
    "Standard_NC48ads_A100_v4"  = @{ Family = "standardNCADSA100v4Family";    VCpus = 48 }
    "Standard_NC40ads_H100_v5"  = @{ Family = "standardNCADSH100v5Family";    VCpus = 40 }
    "Standard_NC16as_T4_v3"     = @{ Family = "Standard NCASv3_T4 Family";    VCpus = 16 }
}
if (-not $sizes.ContainsKey($Size)) { throw "Unknown size '$Size'. Known: $($sizes.Keys -join ', ')" }
$need = $sizes[$Size]

$sub = az account show --query id -o tsv
if (-not $sub) { throw "Not logged in. Run 'az login'." }

Write-Host "== Quota check: $Size ($($need.Family), $($need.VCpus) vCPUs) in $Location"
$raw = az rest --method get `
    --url "https://management.azure.com/subscriptions/$sub/providers/Microsoft.MachineLearningServices/locations/$Location/usages?api-version=2024-10-01" `
    --query "value[?name.value=='$($need.Family)'].{cur:currentValue, limit:limit}" -o json
# -1 limits are per-workspace (uncapped) rows; keep the subscription-level row.
$usages = @($raw | Out-String | ConvertFrom-Json) | ForEach-Object { $_ } |
    Where-Object { $_.limit -ge 0 } | Select-Object -First 1
if (-not $usages) { Write-Warning "Quota family '$($need.Family)' not reported in $Location." }
else {
    $free = $usages.limit - $usages.cur
    Write-Host "   dedicated vCPUs: used $($usages.cur) / limit $($usages.limit) (free $free)"
    if ($free -lt $need.VCpus) {
        Write-Warning ("Insufficient Azure ML quota. Request >= $($need.VCpus) vCPUs for '$($need.Family)' in $Location " +
            "(Azure ML studio > Quota, or Portal > Quotas > Machine learning).")
        if (-not $CheckOnly) { exit 1 }
    }
}
if ($CheckOnly) { exit 0 }

Write-Host "== Workspace $Workspace"
$ws = az ml workspace show -g $ResourceGroup -n $Workspace --query name -o tsv 2>$null
if (-not $ws) {
    az ml workspace create -g $ResourceGroup --file "$here\workspace.yml" --set name=$Workspace location=$Location | Out-Null
}

Write-Host "== Compute instance $ComputeName ($Size)"
$ci = az ml compute show -g $ResourceGroup -w $Workspace -n $ComputeName --query name -o tsv 2>$null
if (-not $ci) {
    az ml compute create -g $ResourceGroup -w $Workspace --file "$here\compute-instance.yml" `
        --set name=$ComputeName size=$Size | Out-Null
}

Write-Host "== Granting compute identity access to $FoundryResource"
$principal = az ml compute show -g $ResourceGroup -w $Workspace -n $ComputeName --query identity.principal_id -o tsv
$scope = az cognitiveservices account show -g $ResourceGroup -n $FoundryResource --query id -o tsv
az role assignment create --assignee-object-id $principal --assignee-principal-type ServicePrincipal `
    --role "Cognitive Services User" --scope $scope | Out-Null

Write-Host "Done. Connect via Azure ML studio / VS Code, clone the repo, then run: bash infra/aml/bootstrap-env.sh"
