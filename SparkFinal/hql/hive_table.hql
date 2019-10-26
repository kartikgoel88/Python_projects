
create external table nse(
SYMBOL string,
SERIES string,
DATE1 string,
PREV_CLOSE double,
OPEN_PRICE double,
HIGH_PRICE double,
LOW_PRICE double,
LAST_PRICE string,
CLOSE_PRICE double,
AVG_PRICE double,
TTL_TRD_QNTY double,
TURNOVER_LACS double,
NO_OF_TRADES double,
DELIV_QTY string ,
DELIV_PER string )
PARTITIONED BY (rundate STRING)
ROW FORMAT DELIMITED
FIELDS TERMINATED BY ','
LINES TERMINATED BY '\n'
STORED AS textFile
LOCATION '/lake/files/';

ALTER TABLE nse ADD PARTITION (rundate=20190914) LOCATION '/lake/files/';
ALTER TABLE nse ADD PARTITION (rundate=20190915) LOCATION '/lake/files/';

--2010-12-01.csv

PARTITIONED BY (rundate STRING)
ROW FORMAT DELIMITED
FIELDS TERMINATED BY ','
LINES TERMINATED BY '\n'
STORED AS textFile
LOCATION '/lake/files/orders/';

LOAD DATA INPATH 'resources/2010-12-01.csv' INTO table orders_hivetable;
ALTER TABLE nse ADD PARTITION (rundate=20190915) LOCATION '/lake/files/orders/';

#overwrite
INSERT OVERWRITE TABLE orders_hivetable PARTITION (rundate=20190915) IF NOT EXISTS
SELECT * FROM orders_hivetable WHERE rundate = 20190914

#append
INSERT INTO TABLE PARTITION (rundate=20190915)
SELECT * FROM orders_hivetable WHERE rundate = 20190914

DROP table orders_hivetable
