$ErrorActionPreference = "Stop"

$repo = "D:\ProgramFiles\Project_4\task_aware_machine_unlearning-master"
$logDir = Join-Path $repo "simulation_result\fed_pipeline_summary\cg_probe_scan_logs"
New-Item -ItemType Directory -Force -Path $logDir | Out-Null

$commands = @(
    "conda run -n unlearning_new python eval_fed_cg_probe.py model=conv criteria=mse +feature_mode=topology_local_fusion +index_mode=helpful +index_criteria=mse +rho=0.001 +num_bus_clients=4 +gnh=true +cg_tol=1e-7 +cg_maxiter=4000 +probe_train_size=128 +probe_test_size=64 +damping=1e-5 +block_damping=1e-5 +result_tag=probe_conv_d1e5",
    "conda run -n unlearning_new python eval_fed_cg_probe.py model=conv criteria=mse +feature_mode=topology_local_fusion +index_mode=helpful +index_criteria=mse +rho=0.001 +num_bus_clients=4 +gnh=true +cg_tol=1e-7 +cg_maxiter=4000 +probe_train_size=128 +probe_test_size=64 +damping=3e-5 +block_damping=3e-5 +result_tag=probe_conv_d3e5",
    "conda run -n unlearning_new python eval_fed_cg_probe.py model=conv criteria=mse +feature_mode=topology_local_fusion +index_mode=helpful +index_criteria=mse +rho=0.001 +num_bus_clients=4 +gnh=true +cg_tol=1e-7 +cg_maxiter=4000 +probe_train_size=128 +probe_test_size=64 +damping=1e-4 +block_damping=1e-4 +result_tag=probe_conv_d1e4",
    "conda run -n unlearning_new python eval_fed_cg_probe.py model=conv criteria=mse +feature_mode=topology_local_fusion +index_mode=helpful +index_criteria=mse +rho=0.001 +num_bus_clients=4 +gnh=true +cg_tol=1e-7 +cg_maxiter=4000 +probe_train_size=128 +probe_test_size=64 +damping=3e-4 +block_damping=3e-4 +result_tag=probe_conv_d3e4",
    "conda run -n unlearning_new python eval_fed_cg_probe.py model=conv criteria=mse +feature_mode=topology_local_fusion +index_mode=helpful +index_criteria=mse +rho=0.001 +num_bus_clients=4 +gnh=true +cg_tol=1e-7 +cg_maxiter=4000 +probe_train_size=128 +probe_test_size=64 +damping=1e-3 +block_damping=1e-3 +result_tag=probe_conv_d1e3",
    "conda run -n unlearning_new python eval_fed_cg_probe.py model=mlpmixer criteria=mse +feature_mode=topology_local_fusion +index_mode=helpful +index_criteria=mse +rho=0.001 +num_bus_clients=4 +gnh=true +cg_tol=1e-7 +cg_maxiter=4000 +probe_train_size=128 +probe_test_size=64 +damping=1e-5 +block_damping=1e-5 +result_tag=probe_mix_d1e5",
    "conda run -n unlearning_new python eval_fed_cg_probe.py model=mlpmixer criteria=mse +feature_mode=topology_local_fusion +index_mode=helpful +index_criteria=mse +rho=0.001 +num_bus_clients=4 +gnh=true +cg_tol=1e-7 +cg_maxiter=4000 +probe_train_size=128 +probe_test_size=64 +damping=3e-5 +block_damping=3e-5 +result_tag=probe_mix_d3e5",
    "conda run -n unlearning_new python eval_fed_cg_probe.py model=mlpmixer criteria=mse +feature_mode=topology_local_fusion +index_mode=helpful +index_criteria=mse +rho=0.001 +num_bus_clients=4 +gnh=true +cg_tol=1e-7 +cg_maxiter=4000 +probe_train_size=128 +probe_test_size=64 +damping=1e-4 +block_damping=1e-4 +result_tag=probe_mix_d1e4",
    "conda run -n unlearning_new python eval_fed_cg_probe.py model=mlpmixer criteria=mse +feature_mode=topology_local_fusion +index_mode=helpful +index_criteria=mse +rho=0.001 +num_bus_clients=4 +gnh=true +cg_tol=1e-7 +cg_maxiter=4000 +probe_train_size=128 +probe_test_size=64 +damping=3e-4 +block_damping=3e-4 +result_tag=probe_mix_d3e4",
    "conda run -n unlearning_new python eval_fed_cg_probe.py model=mlpmixer criteria=mse +feature_mode=topology_local_fusion +index_mode=helpful +index_criteria=mse +rho=0.001 +num_bus_clients=4 +gnh=true +cg_tol=1e-7 +cg_maxiter=4000 +probe_train_size=128 +probe_test_size=64 +damping=1e-3 +block_damping=1e-3 +result_tag=probe_mix_d1e3"
)

$manifest = @()

for ($i = 0; $i -lt $commands.Count; $i++) {
    $idx = "{0:D2}" -f ($i + 1)
    $cmd = $commands[$i]
    $stdout = Join-Path $logDir "$idx.stdout.log"
    $stderr = Join-Path $logDir "$idx.stderr.log"
    $status = "ok"
    try {
        Push-Location $repo
        Invoke-Expression $cmd 1>> $stdout 2>> $stderr
        Pop-Location
    } catch {
        $status = "failed"
        try { Pop-Location } catch {}
        Add-Content -Path $stderr -Value $_.Exception.ToString()
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
