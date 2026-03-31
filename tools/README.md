# Инструменты тестирования

## Быстрый старт

```bash
cd tools
pip install -r requirements.txt

# 1. Сгенерировать тестовое аудио (два голоса: врач + пациент)
python generate_test_audio.py
# → test_calibration.wav (калибровка: ФИО, возраст)
# → test_exam.wav (осмотр: жалобы, анамнез, витальные)
# → test_dialogue.wav (всё вместе)

# 2. Запустить тест через WebSocket (сервер должен быть запущен!)
python ws_streamer.py test_calibration.wav test_exam.wav

# Автоматическая финализация (без ожидания Enter):
python ws_streamer.py test_calibration.wav test_exam.wav --auto-finalize

# Другой сервер:
python ws_streamer.py --url ws://192.168.1.100:8000/ws/session test_calibration.wav test_exam.wav
```

## Что делает generate_test_audio.py

Генерирует WAV файлы с двумя русскими голосами через Microsoft Edge TTS:
- Врач: `ru-RU-DmitryNeural` (мужской)
- Пациент: `ru-RU-SvetlanaNeural` (женский)

Диалог содержит все тестовые кейсы: ФИО, возраст, жалобы, анамнез, витальные показатели,
аллергии, хронические заболевания, наследственность.

## Что делает ws_streamer.py

Эмулирует фронтенд — стримит WAV чанки через WebSocket в реальном времени:
1. Отправляет `start_calibration` + стримит калибровочное аудио + `stop_calibration`
2. Отправляет `start_recording` + стримит аудио осмотра + `stop_recording`
3. Отправляет `finalize`
4. Выводит итоговый протокол

Можно запускать повторно без перезаписи аудио.
