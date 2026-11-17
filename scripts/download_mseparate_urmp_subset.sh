#!/usr/bin/env bash
set -euo pipefail

# Download a small, fixed URMP evaluation subset to an ephemeral runner directory.
# Only mixtures and isolated stems are fetched; videos and the full 12.5 GB dataset are excluded.
# Nothing is committed to Git or uploaded as an audio artifact.

BASE='https://huggingface.co/datasets/Eredis02/URMP/resolve/main'
OUT="${1:-${RUNNER_TEMP:-/tmp}/mseparate-urmp}"
files=(
  '11_Maria_ob_vc/AuMix_11_Maria_ob_vc.wav'
  '11_Maria_ob_vc/AuSep_1_ob_11_Maria.wav'
  '11_Maria_ob_vc/AuSep_2_vc_11_Maria.wav'
  '12_Spring_vn_vn_vc/AuMix_12_Spring_vn_vn_vc.wav'
  '12_Spring_vn_vn_vc/AuSep_1_vn_12_Spring.wav'
  '12_Spring_vn_vn_vc/AuSep_2_vn_12_Spring.wav'
  '12_Spring_vn_vn_vc/AuSep_3_vc_12_Spring.wav'
  '28_Fugue_fl_ob_cl_bn/AuMix_28_Fugue_fl_ob_cl_bn.wav'
  '28_Fugue_fl_ob_cl_bn/AuSep_1_fl_28_Fugue.wav'
  '28_Fugue_fl_ob_cl_bn/AuSep_2_ob_28_Fugue.wav'
  '28_Fugue_fl_ob_cl_bn/AuSep_3_cl_28_Fugue.wav'
  '28_Fugue_fl_ob_cl_bn/AuSep_4_bn_28_Fugue.wav'
  '37_Rondeau_fl_vn_va_cl/AuMix_37_Rondeau_fl_vn_va_cl.wav'
  '37_Rondeau_fl_vn_va_cl/AuSep_1_fl_37_Rondeau.wav'
  '37_Rondeau_fl_vn_va_cl/AuSep_2_vn_37_Rondeau.wav'
  '37_Rondeau_fl_vn_va_cl/AuSep_3_va_37_Rondeau.wav'
  '37_Rondeau_fl_vn_va_cl/AuSep_4_cl_37_Rondeau.wav'
)

for rel in "${files[@]}"; do
  dest="$OUT/$rel"
  mkdir -p "$(dirname "$dest")"
  if [[ -s "$dest" ]]; then
    echo "Already downloaded: $rel"
    continue
  fi
  echo "Downloading: $rel"
  curl --fail --location --retry 4 --retry-all-errors --retry-delay 2 \
    --connect-timeout 30 --max-time 600 "$BASE/$rel" -o "$dest"
  test -s "$dest"
done

(
  cd "$OUT"
  sha256sum -c <<'SUMS'
965d81c53372b00c71b35297ea109266aab728c8127c1b38e9f0c46385e2b4ef  11_Maria_ob_vc/AuMix_11_Maria_ob_vc.wav
28043b76f28ab8170c9a5971ff2a86db43e8b37317d79365a8f846336dc5ee2e  12_Spring_vn_vn_vc/AuMix_12_Spring_vn_vn_vc.wav
6ed0e536afe4a9a0c67bec43b52a8f2060f92ba1f2b7f5625c91d81e2d543079  28_Fugue_fl_ob_cl_bn/AuMix_28_Fugue_fl_ob_cl_bn.wav
f1f753c5e1cd613e2991bdb10dd90ba553430457dd5807a60415f6feebccb13d  37_Rondeau_fl_vn_va_cl/AuMix_37_Rondeau_fl_vn_va_cl.wav
SUMS
)

printf 'Downloaded %s WAV files into %s\n' "${#files[@]}" "$OUT"
