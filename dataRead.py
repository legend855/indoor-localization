from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import itertools

import pandas as pd
import tensorflow as tf

tf.logging.set_verbosity(tf.logging.INFO)

pointsColumns = ["ID", "X", "Y"]
pointsMapping = pd.read_csv("data/PointsMapping.csv", skipinitialspace=True, skiprows=1, names=pointsColumns)

sensColumns = ["timestamp", "AccelerationX", "AccelerationY", "AccelerationZ", "MagneticFieldX", "MagneticFieldY",
                    "MagneticFieldZ", "Z-AxisAgle(Azimuth)", "X-AxisAngle(Pitch)", "Y-AxisAngle(Roll)", "GyroX", "GyroY", "GyroZ"]
timestampColumns = ["Arrival", "Departure", "PlaceID"]

m1PhoneSens = pd.read_csv("data/measure1_smartphone_sens.csv", skipinitialspace=True, skiprows=1, names=sensColumns)
#m1PhoneWifi = pd.read_csv("data/measure1_smartphone_wifi.csv", skipinitialspace=True, names=COLUMNS)
m1WatchSens = pd.read_csv("data/measure1_smartwatch_sens.csv", skipinitialspace=True, skiprows=1, names=sensColumns)
m1TimestampID = pd.read_csv("data/measure1_timestamp_id.csv", skipinitialspace=True, skiprows=1, names=timestampColumns)

m2PhoneSens = pd.read_csv("data/measure2_phone_sens.csv", skipinitialspace=True, skiprows=1, names=sensColumns)
#m2PhoneWifi = pd.read_csv("data/measure2_smartphone_wifi.csv", skipinitialspace=True, names=COLUMNS)
m2WatchSens = pd.read_csv("data/measure2_watch_sens.csv", skipinitialspace=True, skiprows=1, names=sensColumns)
m2TimestampID = pd.read_csv("data/measure2_timestamp_id.csv", skipinitialspace=True, skiprows=1, names=timestampColumns)

# print(m1PhoneSens.as_matrix)
# print(m1WatchSens.as_matrix)
# print(m1TimestampID.as_matrix)
