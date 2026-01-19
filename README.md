1) Клонировать
git clone https://github.com/chyrchxela/NCF.git
cd NCF
2) Виртуальное окружение
python -m venv .venv
source .venv/bin/activate   # macOS / Linux
# .venv\Scripts\activate    # Windows
3) Установить зависимости
pip install -r requirements.txt
4) Обучить модель(если нужно переобучить)
python train_ncf.py
5) Запуск через Docker
docker-compose up --build
6) После запуска REST-API по адресу http://localhost:5001
Веб-интерфейс - по адресу http://localhost:7860
