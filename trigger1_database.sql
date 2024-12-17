-- Create the database if it doesn't already exist
CREATE DATABASE IF NOT EXISTS alerts_db;

-- Use the database
USE alerts_db;

-- Create the table
CREATE TABLE alerts (
    id INT AUTO_INCREMENT PRIMARY KEY,        -- Unique identifier for each alert
    alert_date DATE NOT NULL,                 -- Date of the alert
    alert_time TIME NOT NULL,                 -- Time of the alert
    alert_message VARCHAR(255) NOT NULL,      -- Description of the alert
    longitude DECIMAL(10, 8) NOT NULL,        -- Longitude of the alert location
    latitude DECIMAL(10, 8) NOT NULL,         -- Latitude of the alert location
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP -- Auto-populated timestamp of the record creation
);

-- Display the structure of the table
DESCRIBE alerts;

-- Create a log table to store details about the trigger actions
CREATE TABLE notifications_log (
    id INT AUTO_INCREMENT PRIMARY KEY,
    alert_id INT NOT NULL,                    -- ID of the alert that triggered the action
    log_message VARCHAR(255) NOT NULL,       -- Log message
    triggered_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP -- Time when the trigger was executed
);

INSERT INTO alerts (alert_date, alert_time, alert_message, longitude, latitude)
VALUES ('2024-12-11', '14:30:00', 'Severe weather alert', 77.1025, 28.7041);

SELECT * FROM notifications_log;

DELIMITER //

CREATE TRIGGER after_alert_insert
AFTER INSERT ON alerts
FOR EACH ROW
BEGIN
    -- Insert a log entry into the notifications_log table
    INSERT INTO notifications_log (alert_id, log_message)
    VALUES (
        NEW.id,
        CONCAT('New alert added: "', NEW.alert_message, '" at ', NEW.alert_date, ' ', NEW.alert_time)
    );

    -- Optionally, you can call an external notification API here
    -- You could invoke an external process or procedure if your MySQL environment supports it
    -- For example: SELECT sys_exec('curl -X POST -d "..." http://example.com/api');
END;
//

DELIMITER ;

INSERT INTO alerts (alert_date, alert_time, alert_message, longitude, latitude)
VALUES ('2024-12-11', '14:30:00', 'Severe weather alert', 77.1025, 28.7041);

SELECT * FROM notifications_log;
INSERT INTO alerts (alert_date, alert_time, alert_message, longitude, latitude)
VALUES (CURDATE(), CURTIME(), 'Severe weather alert', 77.1025, 28.7041);

SELECT * FROM notifications_log;
