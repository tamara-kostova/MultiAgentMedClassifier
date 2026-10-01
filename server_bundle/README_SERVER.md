# Упатство за извршување на GPU серверот / Server run instructions

> **Македонски подолу, English version below.**
> Резултатите се запишуваат во **TSV и CSV** фајлови и се пакуваат во една архива на крајот.
> Results are written as **TSV and CSV** files and packed into a single archive at the end.

---

# 🇲🇰 Македонски

## Што е ова

Multi-agent pipeline за автоматска класификација на снимки од мозок (MRI и CT).
Ова е **втората рунда (v2)**: 16 независни експерименти — 4 системи × 4 задачи (Base,
Forest, Debate и хомоген Forest; тумор да/не, тип на тумор, мултиплекс склероза, мозочен
удар). Секој експеримент е **еден Python процес на една GPU** — не користи повеќе јазли и
не бара MPI или Slurm. Секој експеримент ги обработува точно истите 500 слики како
претходната рунда (листите се во `server_bundle/image_lists/`).

Сè се води преку скриптите во `server_bundle/`. Не е потребно да се менува Python код.

## Побарувања

| | |
|---|---|
| GPU | една GPU со ≥ 16 GB VRAM за еден процес (види *Проблеми* ако е помалку) |
| CUDA | 12.6 (сликата е базирана на `nvidia/cuda:12.6.3`) |
| Диск | ~30 GB при отпакување (архиви 14 GB + отпакувано 14 GB); ~15 GB потоа, кога архивите ќе се избришат |
| Софтвер | Singularity или Apptainer |
| Интернет | потребен **само** при градење на сликата, не при извршување |

Сите модели се веќе спакувани во `hf_cache/` — при извршување **нема** пристап до
интернет, нема потреба од HuggingFace токен.

## 1. Отпакување

Двете архиви се отпакуваат во **иста** папка:

```bash
tar xzf maclf-code-data.tar.gz                              # создава MultiAgentMedClassifier/
tar xf  maclf-models.tar -C MultiAgentMedClassifier/        # додава hf_cache/
cd MultiAgentMedClassifier
```

Проверка (треба да ги има сите три):

```bash
ls run_pipeline.py container.def hf_cache data checkpoints
```

## 2. Градење на Singularity сликата

```bash
singularity build --remote container.sif container.def
```

Ако `--remote` не е достапно и има root/fakeroot:

```bash
sudo singularity build container.sif container.def
# или:  singularity build --fakeroot container.sif container.def
```

Градењето трае ~10–20 минути (симнува torch за CUDA 12.6). На крајот печати верзии на
`torch`, `transformers` итн. — ако тоа се испише, сликата е добра.

## 3. Проверки пред долгите извршувања (~1 час) — ВАЖНО

`run_all.sh` и `run_parallel.sh` сами ги пуштаат овие три чекори прво и **не стартуваат
ништо долго** ако 00 или 10 падне. Може да се пуштат и рачно:

| Чекор | Команда | Што прави | Време |
|---|---|---|---|
| 00 | `bash server_bundle/00_preflight.sh` | GPU, податоци, сите 2000 слики од листите, checkpoints, модели; една слика низ Forest и Debate | ~10–20 мин |
| 05 | `bash server_bundle/05_diagnose_cnn.sh` | дијагностика на CNN за тип на тумор (само чита; никогаш не блокира) | ~5 мин |
| 10 | `bash server_bundle/10_smoke.sh` | секој систем на 6 слики, па проверка на секое поле во резултатите | ~45 мин |

- `PREFLIGHT OK` и `SMOKE OK` → може да се стартуваат долгите извршувања;
- `FAILED` → **да не се стартуваат**; ве молам пратете ми ја папката `logs/`.

Чекор 10 печати и **проекција на вкупното време** од измереното време по слика.

## 4. Извршување

### Најпросто — сè по редослед, на една GPU

```bash
nohup bash server_bundle/run_all.sh > logs/run_all.log 2>&1 &
```

Ги извршува проверките, па сите 16 експерименти по редослед. Вкупно ~130 GPU-часа (5–6 дена
на една GPU). Ако еден експеримент падне, скриптата го запишува тоа и продолжува со следниот.

### Ако има повеќе слободни GPU (побрзо)

Експериментите се независни, па може паралелно:

```bash
nohup bash server_bundle/run_parallel.sh 0 1 2 > logs/run_parallel.log 2>&1 &
```

Секој процес користи ~14 GB VRAM. На GPU од 40 GB може ист број да се наведе двапати, на
80 GB до пет пати (пр. `0 0 1 1` = два процеса на секоја од две картички).

### Или експеримент по експеримент, рачно

```bash
bash server_bundle/step.sh <систем>_<задача>
# пр.  bash server_bundle/step.sh debate_stroke
```

| Систем | Што е | Време по задача (500 слики) |
|---|---|---|
| `base` | стандарден pipeline | ~4 ч |
| `forest` | Forest, 4 агенти со различни улоги | ~10 ч |
| `debate` | Debate, 2 рунди | ~12 ч |
| `homog` | хомоген Forest (4 × иста улога, со семплирање), само тријажа | ~6 ч |

Задачи: `binary_tumor`, `multiclass_tumor`, `ms`, `stroke`. Целосниот редослед е во
`server_bundle/steps.sh`. **`multiclass_tumor` е последна намерно** — ако снема време, неа
прескокнете ја прва. Времињата се од друга GPU; точната проекција ја печати чекор 10.

### Прекин, ресетирање, повторно стартување

Секој експеримент е **отпорен на прекин**: резултатот за секоја слика се запишува веднаш.
Ако процес падне, серверот се рестартира или се прекине извршувањето — доволно е **истата
команда да се пушти повторно** и продолжува од сликата на која застанало. Ништо не се
губи и ништо не се повторува.

### Следење на напредокот

```bash
tail -f logs/debate_stroke.log        # logs/<систем>_<задача>.log
tail -f logs/slot*_queue.log          # со run_parallel.sh
```

Секој ред е една слика и содржи предвидување, време и ETA.

## 5. Резултати — што да ми се врати

По завршување:

```bash
bash server_bundle/90_export_results.sh
```

Ова прави архива во главната папка:

```
results_<host>_<датум>.tar.gz
```

Ве молам пратете ми **таа архива**. Ако е преголема за е-пошта, доволна е и само папката
`outputs/results_tsv/`.

Содржина:

| Папка | Што има |
|---|---|
| `outputs/results_tsv/` | **TSV**: еден ред по слика + збирна табела по експеримент + `all_runs_summary.tsv` |
| `outputs/analysis/` | **CSV** табели со метрики (точност, F1, калибрација, confusion matrix) + графици |
| `outputs/eval/v2/` | оригинални JSONL фајлови (целосни детали) |
| `logs/` | логови од извршувањето |

`90_export_results.sh` може да се пушти и **во меѓувреме**, додека траат
експериментите — само чита и не пречи. Така може да се пратат делумни резултати.

## Проблеми

**„CUDA out of memory"** или GPU со помалку од 12 GB
→ во `server_bundle/config.env` поставете `LOAD_4BIT=1` и пуштете ја истата команда
повторно (продолжува од каде застанало).

**`torch.cuda.is_available() is False`**
→ контејнерот мора да се вика со `--nv`; скриптите го прават тоа, значи проблемот е во
драјверот или GPU не е видлива за процесот.

**Слотот е пократок од времето потребно за еден експеримент**
→ во `server_bundle/config.env` намалете `MAX_SAMPLES` (пр. `300`). Резултатите остануваат
валидни, само со помалку слики.

**Друга патека до податоците или до сликата**
→ сите патеки се на едно место: `server_bundle/config.env`. Самите податоци мора да
останат во `data/` како што се отпакувани: листите на слики го содржат токму тој пат до
секоја слика, а preflight го проверува тоа.

**Сè друго**
→ пратете ми го соодветниот лог од `logs/` и продолжувам од тука. Ви благодарам многу!

---

# 🇬🇧 English

## What this is

A multi-agent pipeline for automated classification of brain scans (MRI and CT).
This is the **second round (v2)**: 16 independent experiments, 4 systems × 4 tasks (Base,
Forest, Debate and a homogeneous Forest; tumour yes/no, tumour subtype, multiple sclerosis,
stroke). Each one is a **single Python process on a single GPU** — no multi-node, no MPI,
no Slurm required. Every experiment processes exactly the same 500 images as the previous
round (lists in `server_bundle/image_lists/`).

Everything is driven by the scripts in `server_bundle/`. No Python code needs to be edited.

## Requirements

| | |
|---|---|
| GPU | one GPU with ≥ 16 GB VRAM per process (see *Troubleshooting* for less) |
| CUDA | 12.6 (image is based on `nvidia/cuda:12.6.3`) |
| Disk | ~30 GB while unpacking (14 GB archives + 14 GB extracted); ~15 GB afterwards, once the archives are deleted |
| Software | Singularity or Apptainer |
| Internet | needed **only** to build the image, never at run time |

All model weights are prepacked in `hf_cache/`, so the runs are fully offline and no
HuggingFace token is needed.

## 1. Extract

Both archives extract into the **same** directory:

```bash
tar xzf maclf-code-data.tar.gz                              # creates MultiAgentMedClassifier/
tar xf  maclf-models.tar -C MultiAgentMedClassifier/        # adds hf_cache/
cd MultiAgentMedClassifier
ls run_pipeline.py container.def hf_cache data checkpoints   # all five must exist
```

## 2. Build the image

```bash
singularity build --remote container.sif container.def
# no remote builder? -> sudo singularity build container.sif container.def
```

Takes ~10–20 minutes. It prints the installed `torch` / `transformers` versions at the
end; if you see those, the build is good.

## 3. Checks before the long runs (~1 h) — important

`run_all.sh` and `run_parallel.sh` run these three first by themselves and **start nothing
long** if 00 or 10 fails. They can also be run by hand:

| Step | Command | What it does | Time |
|---|---|---|---|
| 00 | `bash server_bundle/00_preflight.sh` | GPU, datasets, all 2000 listed images, checkpoints, models; one image through Forest and Debate | ~10–20 min |
| 05 | `bash server_bundle/05_diagnose_cnn.sh` | diagnostic of the tumour-subtype CNN (read-only; never blocks) | ~5 min |
| 10 | `bash server_bundle/10_smoke.sh` | every system on 6 images, then a check of every output field | ~45 min |

- `PREFLIGHT OK` and `SMOKE OK` → start the runs.
- `FAILED` → please **do not** start them; send me the `logs/` directory.

Step 10 also prints the **projected total runtime** from the measured seconds/image.

## 4. Run

**One GPU, everything in order** (~130 GPU-hours, 5–6 days):

```bash
nohup bash server_bundle/run_all.sh > logs/run_all.log 2>&1 &
```

**Several GPUs** (~14 GB VRAM per process; list an id twice on a 40 GB card, up to five
times on 80 GB, e.g. `0 0 1 1`):

```bash
nohup bash server_bundle/run_parallel.sh 0 1 2 > logs/run_parallel.log 2>&1 &
```

**Or one experiment at a time:**

```bash
bash server_bundle/step.sh <system>_<task>
# e.g.  bash server_bundle/step.sh debate_stroke
```

| System | What it is | Est. per task (500 images) |
|---|---|---|
| `base` | standard pipeline | ~4 h |
| `forest` | Forest, 4 role-diverse agents | ~10 h |
| `debate` | Debate, 2 rounds | ~12 h |
| `homog` | homogeneous Forest (4 × the same role, sampled), triage only | ~6 h |

Tasks: `binary_tumor`, `multiclass_tumor`, `ms`, `stroke`. The full order is in
`server_bundle/steps.sh`. **`multiclass_tumor` is last on purpose** — skip it first if time
runs out. Estimates come from another GPU; step 10 prints the real projection.

**Interruptions are safe.** Every image is written to disk as soon as it is processed.
If a process dies, the node reboots, or you need the GPU back — just run the same command
again and it continues from where it stopped. Nothing is lost or recomputed.

Follow progress with `tail -f logs/<system>_<task>.log` (one line per image, with ETA), or
`tail -f logs/slot*_queue.log` under `run_parallel.sh`.

## 5. Results to send back

```bash
bash server_bundle/90_export_results.sh
```

creates `results_<host>_<date>.tar.gz` in the project directory — **that archive is what
I need**. If it is too large for email, `outputs/results_tsv/` alone is enough.

| Directory | Contents |
|---|---|
| `outputs/results_tsv/` | **TSV**: one row per image, per-run summaries, `all_runs_summary.tsv` |
| `outputs/analysis/` | **CSV** metric tables (accuracy, F1, calibration, confusion matrices) + plots |
| `outputs/eval/v2/` | raw JSONL (full detail) |
| `logs/` | run logs |

The export script only reads files, so it is safe to run **while runs are still going** if
you want to send partial results early.

## Troubleshooting

| Symptom | Fix |
|---|---|
| `CUDA out of memory`, or GPU < 12 GB | set `LOAD_4BIT=1` in `server_bundle/config.env`, re-run the same command (it resumes) |
| `torch.cuda.is_available() is False` | container must run with `--nv` (the scripts do this) — otherwise a driver/visibility issue |
| Time slot shorter than one run | lower `MAX_SAMPLES` in `server_bundle/config.env` (e.g. `300`) |
| Data or image in a different location | every path lives in `server_bundle/config.env`. The data itself must stay under `data/` as unpacked: the image lists name each file by that path, and preflight checks it |
| Anything else | send me the relevant log from `logs/` — thank you! |
