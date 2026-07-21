# Dependency patches

We fetch each public upstream at a pinned commit and apply a patch containing only the SimFoundry changes.
The `installation_*.sh` scripts do this automatically — you normally don't run anything here by hand.

| Patch | Upstream | Pinned base | Applied by |
|---|---|---|---|
| `samesh-simfoundry.patch` | [gtangg12/samesh](https://github.com/gtangg12/samesh) | `8ec0410` (last upstream commit) | `installation_samesh.sh` |
| `hunyuan-simfoundry.patch` | [Tencent-Hunyuan/Hunyuan3D-Part](https://github.com/Tencent-Hunyuan/Hunyuan3D-Part) | `df0c911` ("switch to safetensors" / PR #15 merge) | `installation_hunyuan.sh` |
| `partfield.patch` | [nv-tlabs/PartField](https://github.com/nv-tlabs/PartField) | default branch | `installation_partfield.sh` |

## How the install scripts use a patch

```bash
git clone <UPSTREAM> <dir>
cd <dir>
git checkout <PIN>
git apply ../../patches/<name>.patch
git submodule update --init --recursive   # SAM2 (samesh); other submodules
```

## What the patches intentionally do NOT contain

- **Model weights** — fetched from HuggingFace at runtime (e.g. `p3sam.safetensors`, SAM2, CoTracker).
- **Large demo/example `.glb` assets** — excluded on purpose; they aren't needed by the pipeline.

## Regenerating a patch

If you change the segmentation code in a cloned+patched checkout, regenerate the patch from the
pinned base so the install stays reproducible:

```bash
# SAMesh — exclude the mistaken MIT LICENSE commit; base is the last upstream commit.
git -C deps/samesh diff 8ec0410 222f388 > patches/samesh-simfoundry.patch

# Hunyuan3D-Part — exclude the large demo .glb so the patch stays text-only.
git -C deps/Hunyuan3D-Part diff df0c911 HEAD -- . \
  ':(exclude)P3-SAM/demo/assets/bamboo_cabinet.glb' \
  ':(exclude)P3-SAM/demo/assets/cabinet.glb' \
  ':(exclude)P3-SAM/demo/assets/laptop.glb' \
  ':(exclude)P3-SAM/demo/assets/open_cabinet.glb' > patches/hunyuan-simfoundry.patch
```

Verify a patch applies against a fresh base checkout before committing it:

```bash
git -C <dir> worktree add -f /tmp/base <PIN>
( cd /tmp/base && git apply --check ../../<repo>/patches/<name>.patch )
git -C <dir> worktree remove --force /tmp/base
```

## Caveats

- **Pin exactly.** Several patched files are near-total rewrites — applying against any commit other
  than the pinned base will reject. `installation_*.sh` checks out the pin before applying.
- **LFS / submodules.** Upstream demo/data assets may be git-LFS (`git lfs pull`) and SAMesh needs
  its SAM2 submodule (`git submodule update --init`); the patch carries neither.
