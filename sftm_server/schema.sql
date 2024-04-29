DROP TABLE IF EXISTS user;
DROP TABLE IF EXISTS traffic;
DROP TABLE IF EXISTS accidents;

CREATE TABLE user(
    id INTEGER PRIMARY KEY AUTOINCREMENT, 
    username TEXT UNIQUE NOT NULL, 
    password TEXT NOT NULL
);

CREATE TABLE traffic(
    date_time TEXT NOT NULL,
    pedestrian_count INTEGER, 
    car_count INTEGER, 
    bicycle_count INTEGER, 
    bus_count INTEGER, 
    motorcycle_count INTEGER, 
    truck_count INTEGER, 
    pedestrian_speed DECIMAL(5, 2), 
    car_speed DECIMAL(5, 2), 
    bicycle_speed DECIMAL(5, 2), 
    bus_speed DECIMAL(5, 2), 
    motorcycle_speed DECIMAL(5, 2), 
    truck_speed DECIMAL(5, 2), 
    volume DECIMAL(5, 2), 
    congestion INTEGER 
);

CREATE TABLE accidents(
    acc_id INTEGER PRIMARY KEY AUTOINCREMENT, 
    date_time TEXT NOT NULL,
    img BLOB NOT NULL, 
    involved TEXT NOT NULL, 
    severity TEXT, 
    stat TEXT NOT NULL
);