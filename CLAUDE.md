# Rules for the agent on this droplet (owner: Viet; read HANDOFF.md next)

## Talking to the owner
- Talk to him plainly and briefly; he is a junior engineer who must be able to explain every piece in an interview.
- Report real numbers, never claims. A step counts as done only when its pass mark is measured.
- Never use the AskUserQuestion tool. Ask in plain text: a short numbered list, each item with a stated default.

## Money
- This MI300X costs $1.99/h on a credit that expires about 15 Oct.
- Billing runs until the droplet is DESTROYED; powering off does not stop it. The owner destroys it; you tell him when.
- Hard stop at 40 alive hours total. Keep the ledger in /root/work/LEDGER.md.
- Do NOT use DigitalOcean snapshots or backups: they are billed to his card, not the credit.

## Secrets and saving work
- Never read, print or log tokens or .env files.
- The Hugging Face login is done by the owner inside the container:
  `docker exec -it sprint hf auth login` (stored under HF_HOME=/work/hf_home).
- Push checkpoints and datasets to his PRIVATE Hugging Face repos at least every 30 minutes. Assume the box can vanish at any time.

## Machine layout
- Work in the docker container `sprint`: `docker exec -it sprint bash`, with /root/work on the host mounted at /work.
  - It has ROCm PyTorch 2.10 and must keep it. ALWAYS install Python packages with `PIP_CONSTRAINT=/work/video2sim-so101/env/constraints.txt` so pip never replaces torch with a CUDA build.
  - lerobot 0.6.1, mujoco 3.11.0.
  - Rendering works only through OSMesa (`MUJOCO_GL=osmesa`); EGL fails here.
- The other container, `rocm`, came with the image and runs a token-protected Jupyter. Leave it alone.

## Subagents
- Pass an explicit model: `sonnet` for research, `opus` for writing code. Never `fable`.
- Run one heavy GPU job at a time unless memory is checked first.

## Physics honesty (this project failed once because of it)
- No welds, attaches or teleporting objects.
- A grasp counts only if both jaws carry the object. That is checked by `task/tests/test_grasp_physics.py` and by the datagen verifier.
