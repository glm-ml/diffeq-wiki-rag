# diffeq-wiki-rag

![CI](https://github.com/glm-ml/diffeq-wiki-rag/actions/workflows/ci.yml/badge.svg)

Agentic RAG-ассистент по англоязычным статьям Википедии о дифференциальных
уравнениях. Отвечает по-русски, цитирует источники и показывает, какие
инструменты использовал агент.

**Демо:** <https://195.62.49.192.nip.io>

## Стек

| Слой | Технологии |
|---|---|
| API и UI | FastAPI, Gradio |
| RAG | LangChain, Qdrant, `intfloat/multilingual-e5-small` |
| Агент | LangGraph, DeepSeek `deepseek-chat` |
| Оценка | RAGAS, LLM-judge, retrieval-метрики |
| Инфраструктура | Docker, nginx, Let's Encrypt, GitHub Actions, GHCR |

## Архитектура

```text
Браузер
  │ HTTPS
  ▼
nginx (TLS, reverse proxy)
  ▼
FastAPI + Gradio
  ├── RAG: e5-small → Qdrant → DeepSeek
  └── Agent: LangGraph → инструменты → DeepSeek
```

### End-to-end запрос

1. Браузер открывает `https://195.62.49.192.nip.io`.
2. DNS `nip.io` возвращает IP сервера.
3. nginx принимает TLS и проксирует запрос в `app:8000`.
4. FastAPI вызывает один из режимов:
   - **RAG** — вопрос кодируется e5-small, Qdrant отдаёт top-5 чанков,
     DeepSeek генерирует ответ по контексту;
   - **агент** — LangGraph сам выбирает инструмент: `search_wikipedia`,
     `run_python` или `web_search`, затем синтезирует ответ.
5. Ответ возвращается с цитатами `[n]`, источниками и, в режиме агента,
   трейсом шагов.

## Данные

- 6 статей Википедии: Lipschitz continuity, Linear differential equation,
  Ordinary differential equation, Picard–Lindelöf theorem, Integrating factor,
  Wronskian.
- 132 чанка после удаления служебных разделов.
- `chunk_id`: `<article>::<section>[::<subsection>]::<chunk_index>`.
- `data/corpus.jsonl` закоммичен для воспроизводимости.

## Golden dataset

| Файл | Содержимое |
|---|---|
| `eval/golden/qa.jsonl` | 30 русских вопросов, эталонные ответы, `key_facts`, `must_cite` |
| `eval/golden/gold_chunks.jsonl` | 60 явных gold-чанков, 1–3 на вопрос |
| `eval/golden/tool_qa.jsonl` | 6 вопросов на выбор инструментов |

Gold-чанки размечены автоматически и вручную; для каждого указан
`chunk_id`, evidence и метод разметки. Схема и валидация — в
[`eval/README.md`](eval/README.md).

## Метрики

Retrieval на 30 вопросах:

| Вариант | recall@5 | section hit@3 | article hit@3 | MRR |
|---|---|---|---|---|
| baseline | 37.2% | 50.0% | 83.3% | 0.787 |
| **combined context** | **65.6%** | **80.0%** | 93.3% | 0.894 |

Ответы и агент:

| Эксперимент | Результат |
|---|---|
| context compaction | контекст −52.1%, correctness 3.73 → 3.50 |
| RAG vs агент | correctness 3.73 → **4.63**, latency 3.1 → 7.9 с |
| tool choice | accuracy **100%**, F1 **1.00**, sequence **100%** |

RAGAS на 30 вопросах (0–1):

| Метрика | Значение |
|---|---|
| faithfulness | **0.893** |
| answer relevancy | 0.782 |
| context precision | 0.578 |
| context recall | 0.675 |
| answer correctness | 0.520 |

RAGAS строже кастомного judge и показывает, что узкое место — retrieval:
ответы заземлены, но top-5 не всегда содержит нужные факты.

Answer-level метрики — LLM-judge по шкале 1–5: correctness, faithfulness,
citations. Эксперименты воспроизводимы в ноутбуках `eval/notebooks/`,
сводка — в [`eval/RESULTS.md`](eval/RESULTS.md).

## Структура

```text
app/
├── agent/        # LangGraph, инструменты, guardrails, трейс
├── ingest/       # сборка корпуса, чанкинг, индексация
├── rag/          # retrieval, промпты, compaction
├── main.py       # FastAPI + Gradio
├── runtime.py    # общее состояние процесса
├── schemas.py    # схемы API
└── ui.py         # Gradio-интерфейс
eval/
├── golden/       # вопросы, gold-чанки, tool-вопросы
├── notebooks/    # 6 экспериментов
├── results/      # JSON с метриками
├── metrics.py    # retrieval-метрики
└── ragas_eval.py # RAGAS-метрики
nginx/            # TLS и reverse proxy
docker/           # entrypoint
tests/            # 57 тестов
Dockerfile
docker-compose.yml
```

## CI/CD

- `ci.yml` — 57 тестов и проверка сборки образа.
- `deploy.yml` — сборка, push в GHCR, SSH-деплой, health-check.
- На сервере образ не собирается: VPS только скачивает готовый образ.
- nginx терминирует TLS, сертификат — Let's Encrypt через `certbot`.
- Qdrant и порт приложения наружу не опубликованы.
