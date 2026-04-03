# Инструменты тестирования

## Быстрый старт

```bash
cd tools
pip install -r requirements.txt

# 1. Сгенерировать тестовое аудио для сценариев (два голоса: врач + пациент)
python generate_test_scenarios.py              # все 11 сценариев
python generate_test_scenarios.py --scenario 01_cardiology  # один
python generate_test_scenarios.py --list        # список доступных

# 2. Запустить тест через WebSocket (сервер должен быть запущен!)
python ws_streamer.py test_scenarios_audio/01_cardiology/calibration.wav test_scenarios_audio/01_cardiology/exam.wav --auto-finalize

# 3. Запустить все сценарии
python run_all_scenarios.py
python run_all_scenarios.py --scenarios 01_cardiology 10_emergency

# 4. Оценить результаты
python evaluate_test.py results/
```

## Файлы

| Файл | Назначение |
|------|-----------|
| `generate_test_scenarios.py` | Генерация WAV из текстовых сценариев через Edge TTS |
| `test_scenarios/scenarios.py` | 11 сценариев: диалоги, голоса, expected-протоколы |
| `ws_streamer.py` | Ручной запуск 1 сценария — стримит WAV на сервер |
| `run_all_scenarios.py` | Массовый запуск сценариев, сохранение результатов |
| `evaluate_test.py` | Расчёт метрик: TC, SA, FER, FVA, DA, OQS, CDS |
| `test_scenarios_audio/` | Сгенерированные WAV для каждого сценария |
| `results/` | JSON с результатами последнего прогона |
