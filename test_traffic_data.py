import sqlite3

def fetch_traffic_information():
    db = sqlite3.connect('instance/sftm.sqlite')
    db.row_factory = sqlite3.Row

    traffic_information = db.execute(
        'SELECT * FROM traffic'
    ).fetchall()

    return traffic_information


if __name__ == "__main__":
    data = fetch_traffic_information()
    print(data)