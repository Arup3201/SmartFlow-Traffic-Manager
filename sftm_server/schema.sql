DROP TABLE IF EXISTS user;
DROP TABLE IF EXISTS camera;

CREATE TABLE user(
    id INTEGER PRIMARY KEY AUTOINCREMENT, 
    username TEXT UNIQUE NOT NULL, 
    password TEXT NOT NULL
);

-- CREATE TABLE camera(
--     c_id INTEGER PRIMARY KEY AUTOINCREMENT,
--     c_name TEXT NOT NULL UNIQUE,
--     yt_id TEXT UNIQUE
-- );

-- INSERT INTO camera (c_name, yt_id) 
-- VALUES 
-- ('Jackson Hole Wyoming USA Town Square Live Cam', '1EiC9bvVGnk'), 
-- ('Sapporo Hokkaido Japan Live 24/7', 'CF1vS8DdBIk'), 
--  ('4 Corners Camera Downtown', 'ByED80IKdIU'), 
-- ('Main Street Livecam, Canmore, Alberta', 'pu6kHxiCSY0');

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
)