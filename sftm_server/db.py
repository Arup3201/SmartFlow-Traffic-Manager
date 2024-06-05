from flask import current_app, g
import sqlite3
import click

CONGESTION_THRED_LOW = [0, 100]
CONGESTION_THRED_MED = [101, 200]

def get_db():
    if 'db' not in g:
        g.db = sqlite3.connect(
            current_app.config['DATABASE'], 
            detect_types=sqlite3.PARSE_DECLTYPES
        )
        g.db.row_factory = sqlite3.Row

    return g.db

def close_db(e=None):
    db = g.pop('db', None)

    if db is not None:
        db.close()

def init_db():
    db = get_db()

    with current_app.open_resource('schema.sql') as f:
        db.executescript(f.read().decode('utf8'))

def clear_initialize_db():
    """Clear the existing data and create new tables."""
    init_db()
    click.echo('Initialized the database.')

def save_to_traffic_db(traffic_info):
    counts = []
    speeds = []
    traffic_volume = 0
    congestion = 0 # 0-> LOW, 1 => MEDIUM 2=> HIGH

    for value in traffic_info.values():
        counts.append(int(value[0]))
        speeds.append(value[1])

    traffic_volume = sum(counts) * (sum(speeds) / 6)

    if CONGESTION_THRED_LOW[0] <= traffic_volume < CONGESTION_THRED_LOW[1]:
        congestion = 0
    elif CONGESTION_THRED_MED[0] <= traffic_volume < CONGESTION_THRED_MED[1]:
        congestion = 1
    else:
        congestion = 2

    db = get_db()

    db.execute(
        "INSERT INTO traffic (date_time, pedestrian_count, car_count, bicycle_count, bus_count, motorcycle_count, truck_count, pedestrian_speed, car_speed, bicycle_speed, bus_speed, motorcycle_speed, truck_speed, volume, congestion) VALUES (datetime('now', 'localtime'), ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)", (traffic_info['person'][0], traffic_info['car'][0], traffic_info['bicycle'][0], traffic_info['bus'][0], traffic_info['motorcycle'][0], traffic_info['truck'][0], traffic_info['person'][1], traffic_info['car'][1], traffic_info['bicycle'][1], traffic_info['bus'][1], traffic_info['motorcycle'][1], traffic_info['truck'][1], traffic_volume, congestion)
    )
    
    db.commit()

def save_to_accident_db(accident_info):
    db = get_db()
    
    query = '''
    INSERT INTO accidents (date_time, img, involved, stat) VALUES (datetime('now'), ?, ?, 'Pending')
    '''
    data_tuple = (accident_info['img'], accident_info['involved'])
    
    try:
        db.execute(query, data_tuple)
        db.commit()
    except db.Error:
        print(f"Error occured when accessing accidents table:\n{db.Error}")

def init_db():
    db = get_db()

    with current_app.open_resource('schema.sql') as f:
        db.executescript(f.read().decode('utf8'))

@click.command('init-db')
def init_db_command():
    """Clear the existing data and create new tables."""
    init_db()
    click.echo('Initialized the database.')

def init_app(app):
    app.teardown_appcontext(close_db)
    app.cli.add_command(init_db_command)