-- products.sql
-- SQL script to create 'products' table and insert 20 product rows

DROP TABLE IF EXISTS products;

CREATE TABLE products (
    product_id TEXT PRIMARY KEY,
    product_name TEXT NOT NULL,
    category TEXT,
    price INTEGER,
    stock INTEGER
);

INSERT INTO products (product_id, product_name, category, price, stock) VALUES ('P001', 'Earphones', 'Electronics', 499, 50);
INSERT INTO products (product_id, product_name, category, price, stock) VALUES ('P002', 'Bluetooth Speaker', 'Audio', 699, 40);
INSERT INTO products (product_id, product_name, category, price, stock) VALUES ('P003', 'Smart Watch', 'Wearable', 1299, 25);
INSERT INTO products (product_id, product_name, category, price, stock) VALUES ('P004', 'Keyboard', 'Computer', 899, 60);
INSERT INTO products (product_id, product_name, category, price, stock) VALUES ('P005', 'Headphones', 'Audio', 1999, 30);
INSERT INTO products (product_id, product_name, category, price, stock) VALUES ('P006', 'Power Bank', 'Mobile Accessories', 2599, 20);
INSERT INTO products (product_id, product_name, category, price, stock) VALUES ('P007', 'USB-C Hub', 'Computer', 1199, 45);
INSERT INTO products (product_id, product_name, category, price, stock) VALUES ('P008', 'Webcam', 'Electronics', 1499, 35);
INSERT INTO products (product_id, product_name, category, price, stock) VALUES ('P009', 'Mouse', 'Computer', 599, 80);
INSERT INTO products (product_id, product_name, category, price, stock) VALUES ('P010', 'Laptop Stand', 'Computer', 799, 55);
INSERT INTO products (product_id, product_name, category, price, stock) VALUES ('P011', 'Neck Band', 'Audio', 849, 65);
INSERT INTO products (product_id, product_name, category, price, stock) VALUES ('P012', 'Smart Band', 'Wearable', 999, 40);
INSERT INTO products (product_id, product_name, category, price, stock) VALUES ('P013', 'LED Desk Lamp', 'Home', 449, 70);
INSERT INTO products (product_id, product_name, category, price, stock) VALUES ('P014', 'Mini Fan', 'Home', 349, 90);
INSERT INTO products (product_id, product_name, category, price, stock) VALUES ('P015', 'Phone Stand', 'Mobile Accessories', 299, 100);
INSERT INTO products (product_id, product_name, category, price, stock) VALUES ('P016', 'Car Charger', 'Mobile Accessories', 399, 75);
INSERT INTO products (product_id, product_name, category, price, stock) VALUES ('P017', 'Screen Protector', 'Mobile Accessories', 199, 120);
INSERT INTO products (product_id, product_name, category, price, stock) VALUES ('P018', 'Cable Organizer', 'Computer', 249, 100);
INSERT INTO products (product_id, product_name, category, price, stock) VALUES ('P019', 'Wireless Charger', 'Electronics', 1099, 30);
INSERT INTO products (product_id, product_name, category, price, stock) VALUES ('P020', 'Portable SSD', 'Computer', 2999, 15);