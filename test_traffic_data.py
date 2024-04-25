import sqlite3

def fetch_traffic_information():
    db = sqlite3.connect('instance/sftm.sqlite')
    db.row_factory = sqlite3.Row

    traffic_information = db.execute(
        'SELECT * FROM traffic'
    ).fetchall()

    data = []
    for ti in traffic_information:
        # data.append(ti['datetime'], ti['pedestrian_count'], ti['bicycle_count'], ti['bus_count'], ti['car_count'], ti['motorcycle_count'], ti['truck_count'], ti['pedestrian_count'], ti['car_speed'], ti['bicycle_speed'], ti['bus_speed'], ti['car_speed'], ti['motorcycle_speed'], ti['truck_speed'], ti['volume'], ti['congestion'])
        data.append(ti['volume'])

    return data

if __name__ == "__main__":
    data = fetch_traffic_information()
    print(data)