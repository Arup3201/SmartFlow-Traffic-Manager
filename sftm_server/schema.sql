DROP TABLE IF EXISTS user;
DROP TABLE IF EXISTS camera;

CREATE TABLE user(
    id INTEGER PRIMARY KEY AUTOINCREMENT, 
    username TEXT UNIQUE NOT NULL, 
    password TEXT NOT NULL
);

CREATE TABLE camera(
    c_id INTEGER PRIMARY KEY AUTOINCREMENT,
    c_name TEXT NOT NULL UNIQUE,
    yt_link TEXT UNIQUE
);

INSERT INTO camera (c_name, yt_link) 
VALUES 
('Jackson Hole Wyoming USA Town Square Live Cam', 'https://www.youtube.com/watch?v=1EiC9bvVGnk'), 
('Sapporo Hokkaido Japan Live 24/7', 'https://www.youtube.com/watch?v=CF1vS8DdBIk'), 
 ('4 Corners Camera Downtown', 'https://www.youtube.com/watch?v=ByED80IKdIU'), 
('Main Street Livecam, Canmore, Alberta', 'https://www.youtube.com/watch?v=pu6kHxiCSY0');