$ErrorActionPreference = "Stop"
$models = @("unet", "fno", "afno", "itransolver", "pod-fno", "pod-afno", "pod-unet", "pod-itransformer", "pod-itransolver")
foreach ($model in $models) {
    python -m pod_sim2real.training.train --config configs/smoke.yaml --model $model
}
