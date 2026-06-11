$ErrorActionPreference = "Stop"

$repo = "D:\ProgramFiles\Project_4\task_aware_machine_unlearning-master"
$logDir = Join-Path $repo "simulation_result\fed_pipeline_summary\conv_topology_ablation_logs"
New-Item -ItemType Directory -Force -Path $logDir | Out-Null

$commands = @(
    "conda run -n unlearning_new python eval_fed_unchange.py model=conv criteria=mse +feature_mode=topology_local_fusion +index_mode=helpful +index_criteria=mse +rho=0.001 +num_bus_clients=4 +gnh=true +cg_tol=1e-7 +cg_maxiter=4000 +damping=3e-4 +block_damping=3e-4 +topology_partition_mode=topology +encoder_topology_mode=topology +fusion_topology_alpha=0.5 +result_tag=abl_full",
    "conda run -n unlearning_new python eval_fed_unchange.py model=conv criteria=mse +feature_mode=topology_local_fusion +index_mode=helpful +index_criteria=mse +rho=0.001 +num_bus_clients=4 +gnh=true +cg_tol=1e-7 +cg_maxiter=4000 +damping=3e-4 +block_damping=3e-4 +topology_partition_mode=topology +encoder_topology_mode=topology +fusion_topology_alpha=0 +result_tag=abl_part",
    "conda run -n unlearning_new python eval_fed_unchange.py model=conv criteria=mse +feature_mode=topology_local_fusion +index_mode=helpful +index_criteria=mse +rho=0.001 +num_bus_clients=4 +gnh=true +cg_tol=1e-7 +cg_maxiter=4000 +damping=3e-4 +block_damping=3e-4 +topology_partition_mode=uniform +encoder_topology_mode=local_only +fusion_topology_alpha=0 +result_tag=abl_none"
)

$manifest = @()

for ($i = 0; $i -lt $commands.Count; $i++) {
    $idx = "{0:D2}" -f ($i + 1)
    $cmd = $commands[$i]
    $stdout = Join-Path $logDir "$idx.stdout.log"
    $stderr = Join-Path $logDir "$idx.stderr.log"
    $status = "ok"
    $warningOnly = $true

    try {
        Push-Location $repo
        & powershell -NoProfile -Command $cmd 1>> $stdout 2>> $stderr
        $exitCode = $LASTEXITCODE
        Pop-Location
        if ($exitCode -ne 0) {
            $status = "failed"
            $warningOnly = $false
        }
    } catch {
        $status = "failed"
        try { Pop-Location } catch {}
        Add-Content -Path $stderr -Value $_.Exception.ToString()
        $warningOnly = $false
    }

    if (Test-Path $stderr) {
        $stderrText = Get-Content $stderr -Raw
        if ($stderrText.Trim().Length -gt 0) {
            $normalized = ($stderrText -replace "`r","" -replace "`n"," ").Trim()
            if ($normalized -notmatch "Defaults list is missing `_self_") {
                $warningOnly = $false
            }
        }
    }

    if ($status -eq "ok" -and -not $warningOnly) {
        $status = "warning"
    }

    $manifest += [pscustomobject]@{
        order = $idx
        status = $status
        command = $cmd
        stdout_log = $stdout
        stderr_log = $stderr
        finished_at = (Get-Date).ToString("yyyy-MM-dd HH:mm:ss")
    }
    $manifest | Export-Csv -NoTypeInformation -Path (Join-Path $logDir "manifest.csv")
}
