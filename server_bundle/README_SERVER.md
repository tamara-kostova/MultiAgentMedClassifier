# Упатство за извршување на GPU серверот / Server run instructions

> **Македонски подолу, English version below.**
> Резултатите се запишуваат во **TSV и CSV** фајлови и се пакуваат во една архива на крајот.
> Results are written as **TSV and CSV** files and packed into a single archive at the end.

---

# 🇲🇰 Македонски

## Што е ова

Multi-agent pipeline за автоматска класификација на снимки од мозок (MRI и CT).
Треба да се извршат **шест независни експерименти**. Секој експеримент е **еден Python
процес на една GPU** — не користи повеќе јазли и не бара MPI или Slurm.

Сè се води преку нумерирани скрипти во `server_bundle/`. Не е потребно да се менува
Python код.

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

## 3. Проверка пред долгите извршувања (~10–20 мин) — ВАЖНО

```bash
bash server_bundle/00_preflight.sh
```

Оваа скрипта проверува GPU, податоците, checkpoints и моделите, и потоа пушта **една
слика** низ двата pipeline-а (Forest и Debate). На крај печати:

- `PREFLIGHT OK` → може да се стартуваат долгите извршувања;
- `PREFLIGHT FAILED` → **да не се стартуваат**; ве молам пратете ми
  `logs/00_preflight.log`.

Печати и измерено **време по слика**, па веднаш се знае колку ќе трае секој експеримент.

## 4. Извршување

### Најпросто — сè по редослед, на една GPU

```bash
nohup bash server_bundle/run_all.sh > logs/run_all.log 2>&1 &
```

Тоа ги извршува сите чекори по редослед. Вкупно ~3–4 дена на една GPU.
Ако еден чекор падне, скриптата го запишува тоа и продолжува со следниот.

### Ако има повеќе слободни GPU (побрзо)

Експериментите се независни, па може паралелно — по еден процес на GPU:

```bash
nohup bash server_bundle/run_parallel.sh 0 1 2 > logs/run_parallel.log 2>&1 &
```

Со три GPU сè завршува во ~1.5 ден. Секој процес користи ~14 GB VRAM.
(На GPU од 40/80 GB може ист број да се наведе двапати, пр. `0 0 1 1`, за два процеса на
таа картичка.)

### Или чекор по чекор, рачно

| Ред | Команда | Задача | Приближно време (500 слики) |
|---|---|---|---|
| 1 | `bash server_bundle/01_forest_stroke.sh` | Forest N=4, мозочен удар (CT) | ~8 ч |
| 2 | `bash server_bundle/02_forest_ms.sh` | Forest N=4, мултиплекс склероза | ~8 ч |
| 3 | `bash server_bundle/03_debate_binary_tumor.sh` | Debate R=2, тумор да/не | ~13–16 ч |
| 4 | `bash server_bundle/04_debate_stroke.sh` | Debate R=2, мозочен удар | ~12–16 ч |
| 5 | `bash server_bundle/05_debate_ms.sh` | Debate R=2, мултиплекс склероза | ~12–16 ч |
| 6 | `bash server_bundle/06_debate_multiclass_tumor.sh` | Debate R=2, тип на тумор | ~14–16 ч |

Времињата се проценки од друга (послаба) GPU — точната проекција ја печати чекор 3.
**Чекор 6 е последен намерно**: ако снема време, тој е најмалку важен и може да се прескочи.

### Прекин, ресетирање, повторно стартување

Секој експеримент е **отпорен на прекин**: резултатот за секоја слика се запишува веднаш.
Ако процес падне, серверот се рестартира или се прекине извршувањето — доволно е **истата
команда да се пушти повторно** и продолжува од сликата на која застанало. Ништо не се
губи и ништо не се повторува.

### Следење на напредокот

```bash
tail -f logs/01_forest_stroke.log        # или друг чекор
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
| `outputs/eval/` | оригинални JSONL фајлови (целосни детали) |
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
→ сите патеки се на едно место: `server_bundle/config.env`.

**Сè друго**
→ пратете ми го соодветниот лог од `logs/` и продолжувам од тука. Ви благодарам многу!

---

# 🇬🇧 English

## What this is

A multi-agent pipeline for automated classification of brain scans (MRI and CT).
There are **six independent experiments**. Each one is a **single Python process on a
single GPU** — no multi-node, no MPI, no Slurm required.

Everything is driven by the numbered scripts in `server_bundle/`. No Python code needs
to be edited.

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

## 3. Preflight (~10–20 min) — please run this first

```bash
bash server_bundle/00_preflight.sh
```

Checks the GPU, datasets, checkpoints and model cache, then pushes **one image** through
both pipelines (Forest and Debate) and prints the measured seconds/image, so the runtime
of the full runs is known before committing GPU time.

- `PREFLIGHT OK` → start the runs.
- `PREFLIGHT FAILED` → please **do not** start them; send me `logs/00_preflight.log`.

## 4. Run

**One GPU, everything in order** (~3–4 days total):

```bash
nohup bash server_bundle/run_all.sh > logs/run_all.log 2>&1 &
```

**Several GPUs** (~1.5 days on three; ~14 GB VRAM per process):

```bash
nohup bash server_bundle/run_parallel.sh 0 1 2 > logs/run_parallel.log 2>&1 &
```

**Or step by step:**

| # | Command | Experiment | Est. (500 images) |
|---|---|---|---|
| 1 | `bash server_bundle/01_forest_stroke.sh` | Forest N=4, stroke CT | ~8 h |
| 2 | `bash server_bundle/02_forest_ms.sh` | Forest N=4, multiple sclerosis | ~8 h |
| 3 | `bash server_bundle/03_debate_binary_tumor.sh` | Debate R=2, tumour yes/no | ~13–16 h |
| 4 | `bash server_bundle/04_debate_stroke.sh` | Debate R=2, stroke CT | ~12–16 h |
| 5 | `bash server_bundle/05_debate_ms.sh` | Debate R=2, multiple sclerosis | ~12–16 h |
| 6 | `bash server_bundle/06_debate_multiclass_tumor.sh` | Debate R=2, tumour subtype | ~14–16 h |

Estimates come from a weaker GPU; step 3 prints the real projection. **Step 6 is last on
purpose** — it is the least informative, so skip it first if time runs out.

**Interruptions are safe.** Every image is written to disk as soon as it is processed.
If a process dies, the node reboots, or you need the GPU back — just run the same command
again and it continues from where it stopped. Nothing is lost or recomputed.

Follow progress with `tail -f logs/<step>.log` (one line per image, with ETA).

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
| `outputs/eval/` | raw JSONL (full detail) |
| `logs/` | run logs |

The export script only reads files, so it is safe to run **while runs are still going** if
you want to send partial results early.

## Troubleshooting

| Symptom | Fix |
|---|---|
| `CUDA out of memory`, or GPU < 12 GB | set `LOAD_4BIT=1` in `server_bundle/config.env`, re-run the same command (it resumes) |
| `torch.cuda.is_available() is False` | container must run with `--nv` (the scripts do this) — otherwise a driver/visibility issue |
| Time slot shorter than one run | lower `MAX_SAMPLES` in `server_bundle/config.env` (e.g. `300`) |
| Data or image in a different location | every path lives in `server_bundle/config.env` |
| Anything else | send me the relevant log from `logs/` — thank you! |
