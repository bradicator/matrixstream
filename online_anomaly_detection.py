#!/home/mgkallit/.local/share/canopy/edm/envs/User/bin/python
import time as time_t
import numpy as np
from scipy import linalg
import csv
import sys
import logging
from pymongo import MongoClient

# *************************************************************************** #
# ===================== Configuration Parameters ============================ #
# *************************************************************************** #

# ********************* Parameters of the process *************************** #

# Dimensions for the process
BRICK_DIMENSION = 128

# How many IP hash bins are reserved for known servers
NUMBER_OF_BINS_FOR_KNOWN_SERVERS = 8

# ********************** Monitoring parameters ****************************** #

# Time interval for monitoring updates,
# should be roughly the same as interval of database updates
# recommended: 3 (unit: seconds)
TIME_INTERVAL_TO_UPDATE = 3

# The number of databricks to warm-start the monitoring,
# recommended: 2 to 10 * BRICK_DIMENSION (unitless)
STARTING_BATCH_SIZE = BRICK_DIMENSION * 2

# Width of control limits
# recommended: 10  (unit: normal quantile)
L_CTRL_LMT_MULTIPLE = 5

# level above which observations will not be used for parameter estimation,
# recommended: 5 (unit: normal quantile)
ROBUST_ESTIMATION_GAURD = 5

# How long of an effective 'memory' should PCA estimations keep?
# recommended 600,000 / TIME_INTERVAL_TO_UPDATE (roughly 7 days)
# (unit: seconds / seconds)
MAX_PCA_MEMORY = 600000 / TIME_INTERVAL_TO_UPDATE

# AR(p) model size, used to whiten residual series
# recommended: < 5
ARP_SIZE = 5

# How much should the current observation affect the control parameter
# estimations for whitening residuals?
# recommended: 0.005 or smaller
PARAMETER_ESTIMATION_SMOOTHING = 0.005

# How long of an interval should two alerts be apart to be considered separate
# events?
# Recommended 60 (unit: seconds)
SAME_EVENT_TIME_BUFFER = 60

# ************************* mongoDB parameters ****************************** #

# db, port, collection names
MONGO_HOST = '198.108.63.80'
PORT = 27017
DB = 'amon'
READ_COLLECTION = 'databrick_equinix'
WRITE_COLLECTION = 'alerts_anomaly_detection'
ONGOING_EVENT_COLLECTION = 'ongoing_events'

# Fields for writing the output to db
FIELD_NAME_EVENT_ID = 'event_ID'
FIELD_NAME_BIN_NUMBER = 'bin_number'
FIELD_NAME_START = 'start'
FIELD_NAME_END = 'end'
FIELD_NAME_DURATION = 'duration'
FIELD_NAME_INTENSITY = 'intensity'

# Diagnostics database
DIAGNOSTICS_DB = 'amon_diagnostics'
PCA_ESTIMATIONS_COLLECTION = 'PCA_estimations'
CONFIGURATIONS_COLLECTION = 'configurations'
RESIDUAL_CONTROLS_COLLECTION = 'residual_controls'

# Output decimal places
DIGIT_ROUNDING = 3


# *************************************************************************** #
# ========================== Utility functions  ============================= #
# *************************************************************************** #

# ======== Source and destination-wise sums and sums of logarithms  ========= #


def preprocessing(collection, number_of_records):
    """
    This function computes the sum and sum of logarithms a databrick,
    both source-wise and destination-wise. Records are retrieved in
    reverse chronological order.
    :param collection: The mongoBD database name, string
    :param number_of_records: the number of most recent reocords retrieved, scalar
    :return: raw traffic volume by source (number_of_records by p),
             sum of logarithms of traffic volume by source (number_of_records by p),
             raw traffic volume by destination (number_of_records by p),
             sum of logarithms of traffic volume by detination (number_of_records by p),
             time stamps of the records, (1 by number_of_records)
    """
    sum_of_dst_databricks = []
    logaritmic_sum_of_dst_databricks = [] 
    sum_of_src_databricks = []
    logaritmic_sum_of_src_databricks = []
    time = []

    for brick in collection.find().sort("timestamp", -1).limit(number_of_records):

        try:
            databrick = np.array(brick["data"])
            col_wise_databricks = np.reshape(databrick, (BRICK_DIMENSION, BRICK_DIMENSION), order='F')
            col_wise_databricks = col_wise_databricks[NUMBER_OF_BINS_FOR_KNOWN_SERVERS:BRICK_DIMENSION,
                                                      NUMBER_OF_BINS_FOR_KNOWN_SERVERS:BRICK_DIMENSION]
            ln_of_col_wise_databricks = np.log1p(col_wise_databricks)
            sum_of_dst_databricks.append(col_wise_databricks.sum(axis=1))
            logaritmic_sum_of_dst_databricks.append(ln_of_col_wise_databricks.sum(axis=1))
            sum_of_src_databricks.append(col_wise_databricks.sum(axis=0))
            logaritmic_sum_of_src_databricks.append(ln_of_col_wise_databricks.sum(axis=0))
            time.append(int(time_t.mktime(time_t.strptime(str(brick["timestamp"]), '%Y-%m-%d %H:%M:%S'))))
        except Exception:
            logging.exception("Bad Data at %s", str(brick["timestamp"]))

    return (np.asanyarray(sum_of_src_databricks),
            np.asanyarray(logaritmic_sum_of_src_databricks),
            np.asanyarray(sum_of_dst_databricks),
            np.asanyarray(logaritmic_sum_of_dst_databricks),
            time)


# ============ This function does sequential estimation of PC's ============= #


def incremental_PCA(first_k_eigenvalues, first_k_eigenvectors, x, n, center, tol=1e-07):
    """
    This function does sequential estimation of PC's
    :type first_k_eigenvalues: np.array
    :param first_k_eigenvalues: the inital estimate of eignevalues, 1 by k
    :param first_k_eigenvectors: the initial estimate of eigenvectors, p by k
    :param x: the new data point, 1 by p
    :param n: the effective number of data points up to now, recommend capping this number
    :param center: the estimated mean of the data, 1 by p
    :param tol: tolerance
    :return: Updated first_k_eigenvalues and first_k_eigenvectors
    """
    q = len(first_k_eigenvalues)
    f = 1 / n
    k = len(first_k_eigenvalues)
    assert first_k_eigenvectors.shape[1] == k, "length(lambda) != ncol(U)"
    assert first_k_eigenvectors.shape[0] == len(x), "length(lambda) != ncol(U)"
    x = x - center
    x = np.sqrt(f) * x
    xhat = np.dot(first_k_eigenvectors.transpose(), x)
    x = x - np.dot(first_k_eigenvectors, xhat)
    normx = np.linalg.norm(x)

    if normx >= tol:
        k = k + 1
        first_k_eigenvalues = np.append(first_k_eigenvalues, [0])
        xhat = np.append(xhat, normx)
        first_k_eigenvectors = np.column_stack((first_k_eigenvectors, x / normx))

    first_k_eigenvalues, eigvec = np.linalg.eigh(np.diag(first_k_eigenvalues) + np.outer(xhat, xhat))
    idx = first_k_eigenvalues.argsort()[::-1]
    first_k_eigenvalues = first_k_eigenvalues[idx]
    eigvec = eigvec[:, idx]

    if q < k:
        first_k_eigenvalues = first_k_eigenvalues[:q]
        eigvec = eigvec[:, range(0, q)]
    return first_k_eigenvalues * (1 - f), np.dot(first_k_eigenvectors, eigvec)


# ========= This function does sequential estimation of mean vector ========= #

def update_mean(center, x, mean_smoothing):
    """
    This function does sequential estimation of mean vector
    :type x: numpy.array
    :type mean_smoothing: float
    :type center: numpy.array
    :param center: initial estimate of mean vector, 1 by p
    :param x: new data point, 1 by p
    :param mean_smoothing: scalar, weight placed on the current observation
    :return: updated mean estimate
    """
    center = (1 - mean_smoothing) * center + mean_smoothing * x
    return center


# ========= The sample auto-covarince function, 1/n not 1/(n-1) =========== #

def acf(data, max_lag=20):
    """
    calculates the auto-covariance function up to the maximum number of lags,
    lag 0 being the marginal variance.
    :param data: the data series, 1-dimensional time-series
    :param max_lag: maximum number of lags to calculate
    :return: a list of auto-covariances, length (max_laf + 1)
    """
    n = data.size
    return [np.mean(data[lag:n]*data[0:(n-lag)]) for lag in range(0, (max_lag+1))]


# *************************************************************************** #
# ====== This is a class that maintains the sequential estimations ========== #
# ====== It maintains the estimation of first k eigenvalues and    ========== #
# ====== eigenvectors using Arora's incremental PCA algorithm (2012) ======== #
# *************************************************************************** #

class StreamPcaEstimations(object):

    __slots__ = ["first_k_eigenvalues", "first_k_eigenvectors", "center", "variance_explained",
                 "effective_number_of_data_count", "k"]

    def __init__(self, x, top_k=5):
        """Initializing the tracker with a small batch data"""
        x = np.array(x)
        x = x.astype(np.float)

        self.center = np.mean(x, axis=0)
        x = x - np.outer(np.ones(ACTUAL_STARTING_BATCH_SIZE), self.center)

        self.first_k_eigenvalues, self.first_k_eigenvectors = np.linalg.eigh(np.cov(x.transpose()))
        # have to sort the eigen values by magnitude
        idx = self.first_k_eigenvalues.argsort()[::-1]
        self.first_k_eigenvalues = self.first_k_eigenvalues[idx]
        self.first_k_eigenvectors = self.first_k_eigenvectors[:, idx]

        # remember the first top 2*k eigenvalues / eigenvectors
        self.k = top_k
        self.first_k_eigenvalues = self.first_k_eigenvalues[range(0, 2*self.k)]
        self.first_k_eigenvectors = self.first_k_eigenvectors[:, range(0, 2*self.k)]

        # calculate variance explained
        self.variance_explained = sum(self.first_k_eigenvalues[range(0, self.k)]) / sum(self.first_k_eigenvalues)

        self.effective_number_of_data_count = x.shape[0]

    def initial_residuals(self, x):
        """
        Calculate residuals for the initial batch data.
        Should only be called after instance initialization.
        :param x: the initial batch of data, dimension: ACTUAL_STARTING_BATCH_SIZE by p
        :return: the initial batch of residuals, dimension: ACTUAL_STARTING_BATCH_SIZE by p
        """
        x = np.array(x)
        x = x.astype(np.float)
        x = x - np.outer(np.ones(ACTUAL_STARTING_BATCH_SIZE), self.center)
        xhat = np.dot(x, self.first_k_eigenvectors[:, 0:self.k])
        xhat = np.dot(xhat, self.first_k_eigenvectors[:, 0:self.k].transpose())
        initial_batch_residuals = x - xhat
        return initial_batch_residuals

    def update(self, x):
        """
        updates parameter estimates of PCA: principal components and means
        :type x: np.array
        :param x: a new observation, 1 by p
        :return: residuals, 1 by p
        """
        x = np.reshape(x, BRICK_DIMENSION - NUMBER_OF_BINS_FOR_KNOWN_SERVERS)
        self.center = update_mean(self.center, x,
                                  mean_smoothing=max(1 / self.effective_number_of_data_count, 1 / MAX_PCA_MEMORY))
        self.first_k_eigenvalues, self.first_k_eigenvectors = \
            incremental_PCA(self.first_k_eigenvalues, self.first_k_eigenvectors, x,
                            n=min(self.effective_number_of_data_count, MAX_PCA_MEMORY), center=self.center)
        x = x - self.center
        xhat = np.dot(self.first_k_eigenvectors[:, 0:self.k].transpose(), x)
        pca_residuals = x - np.dot(self.first_k_eigenvectors[:, 0:self.k], xhat)

        # update variance explained
        self.variance_explained = update_mean(self.variance_explained, np.linalg.norm(xhat) / np.linalg.norm(x),
                                  mean_smoothing=max(1 / self.effective_number_of_data_count, 1 / MAX_PCA_MEMORY))

        self.effective_number_of_data_count = self.effective_number_of_data_count + 1

        return pca_residuals


# *************************************************************************** #
# ====== This is a class that maintains the monitoring parameters for ======= #
# ====== for *one* time series.                                       ======= #
# ====== It maintains a short memory for residuals and whitened series ====== #
# ====== as well as estimated auto-covariance, AR(p) coefficients for ======= #
# ====== residual time series, and estimated marginal variance for the ====== #
# ====== whietened time series.                                       ======= #
# *************************************************************************** #

class StreamMonitor(object):

    __slots__ = ['residuals_recent_few', 'ARp_size', 'residual_ewma', 'param_smoothing',
                 'residual_maringal_variance', 'residual_autocov', 'ARp_coeff',
                 'whitened', 'whitened_mar_var',
                 'EWMA', 'ewma_smoothing']

    def __init__(self, initial_batch_residuals,
                 arp_size=ARP_SIZE,
                 parameter_estimation_smoothing=PARAMETER_ESTIMATION_SMOOTHING,
                 ewma_smoothing=0.1):
        """
        Initializing the monitor with a small batch data
        :param initial_batch_residuals: the initial dataset of residuals, ACTUAL_STARTING_BATCH_SIZE by p
        :param arp_size: size of AR model
        :param parameter_estimation_smoothing: smoothing parameter for parameter estimations
        :param ewma_smoothing: smoothing parameter for whitened residuals EWMA smoothing
        """
        # The parameters that are read in directly
        self.ARp_size = arp_size
        # most recent residuals, from current (indexed at 0) to
        # the the least recent one (indexed at ARp_size)
        self.residuals_recent_few = list(reversed(initial_batch_residuals[-self.ARp_size:]))
        self.param_smoothing = parameter_estimation_smoothing

        # The parameters that needs simple estimations
        self.residual_ewma = np.mean(initial_batch_residuals)
        self.residual_maringal_variance = np.mean(initial_batch_residuals * initial_batch_residuals)
        self.residual_autocov = acf(initial_batch_residuals, max_lag=self.ARp_size)

        # Solving Yule-Walker equations for ARp coefficients
        residual_autocorr = [i / self.residual_maringal_variance for i in self.residual_autocov]
        R = linalg.toeplitz(residual_autocorr[:-1])
        self.ARp_coeff = np.dot(np.linalg.inv(R), residual_autocorr[1:])

        # Whiten residual series
        ARp_filter = np.insert([-i for i in self.ARp_coeff], 0, [1])
        self.whitened = np.convolve(initial_batch_residuals, ARp_filter, mode='valid')
        self.whitened_mar_var = np.mean(self.whitened * self.whitened)

        # Whitened EWMA stuff
        self.EWMA = np.mean(self.whitened)
        self.ewma_smoothing = ewma_smoothing

    def update(self, x):
        """
        updates memory and parameter estimates
        :param x: the new observation in the stream, scalar
        :return: nothing
        """
        # increase memory size by 1, x has to be saved before we estimate the AR coefficients
        self.residuals_recent_few = np.insert(self.residuals_recent_few, 0, x)
        # removing slow EWMA from residuals
        if np.absolute(x) < (ROBUST_ESTIMATION_GAURD * np.sqrt(self.residual_maringal_variance)):
            self.residual_ewma = (1 - self.param_smoothing) * self.residual_ewma + self.param_smoothing * x
        x = x - self.residual_ewma

        # robust variance/covariance updates
        if np.absolute(x) < (ROBUST_ESTIMATION_GAURD * np.sqrt(self.residual_maringal_variance)):
            for j in range(self.ARp_size + 1):
                self.residual_autocov[j] = (1 - self.param_smoothing) * self.residual_autocov[j] + \
                                           self.param_smoothing * self.residuals_recent_few[j] * x
            self.residual_maringal_variance = self.residual_autocov[0]

        # Solving Yule-Walker equations
        residual_autocorr = [i / self.residual_maringal_variance for i in self.residual_autocov]
        R = linalg.toeplitz(residual_autocorr[:-1])
        self.ARp_coeff = np.dot(np.linalg.inv(R), residual_autocorr[1:])

        # Whiten residual series
        ARp_filter = np.insert([-i for i in self.ARp_coeff], 0, [1])
        self.whitened = np.convolve(self.residuals_recent_few, ARp_filter, mode='valid')
        self.whitened_mar_var = (1 - self.param_smoothing) * self.whitened_mar_var + \
            self.param_smoothing * self.whitened * self.whitened

        # reduce memory size by 1
        self.residuals_recent_few = np.delete(self.residuals_recent_few, self.ARp_size)

        # Whitened EWMA stuff
        self.EWMA = (1 - self.ewma_smoothing) * self.EWMA + self.ewma_smoothing * self.whitened

    def alerts(self, L):
        """
        raise alerts if process falls outside control limits
        :param L: Control limit multiple
        :return: alert location(s), and severity(s)
        """
        simple_alert = np.absolute(self.residuals_recent_few[-1]) > (np.sqrt(self.residual_maringal_variance) * L)
        simple_intensity = self.residuals_recent_few[-1] / np.sqrt(self.residual_maringal_variance)
        ewma_alert = np.absolute(self.EWMA) > (np.sqrt(self.whitened_mar_var *
                                          (self.ewma_smoothing / (2 - self.ewma_smoothing))) * L)
        ewma_intensity = self.EWMA / np.sqrt(self.whitened_mar_var *
                                             (self.ewma_smoothing / (2 - self.ewma_smoothing)))
        return simple_alert, simple_intensity, ewma_alert, ewma_intensity


# *************************************************************************** #
# ====== The Event Recorder class                                    ======== #
# ====== One instance is maintained for each stream                  ======== #
# ====== It maintains the starting time and end time of an event     ======== #
# ====== If the event is on-going, a binary flag is set to indicate  ======== #
# ====== 'severity' is reserved for later use                        ======== #
# *************************************************************************** #

class EventRecorder(object):

        __slots__ = ['event_id', 'bin_number', 'start', 'end', 'duration',
                     'ongoing', 'last_alert',
                     'avg_intensity', 'collection_to_write_to', 'collection_for_ongoing_events']

        def __init__(self, bin, write_collection, ongoing_collection):
            """
            Initialize the Event Recorder
            :param bin: bin number of the IP addresses
            :param path_to_file: Path of the output file to write to
            """
            self.bin_number = bin
            self.collection_to_write_to = write_collection
            self.collection_for_ongoing_events = ongoing_collection
            self.ongoing = False

        def create_id(self):
            """
            Generate event ID, used internally
            :return:
            """
            # Event ID is the integer index of the number of events in the database
            result = self.collection_to_write_to.count() + self.collection_for_ongoing_events.count()
            self.event_id = result

        def update(self, alert, intensity, time):
            """
            Update the event recorder with latest alert. Warning: tricky logic
            :param alert: Boolean value, whether an alert was raised
            :param intensity: the intensity of current traffic, in normal quantiles
            :param time: time of current update
            :return: nothing
            """
            # when there is an alert,
            if alert == True:
                # check if it a new event
                if self.ongoing == False:
                    # records time in epoch time
                    self.start = time
                    # records event id
                    self.create_id()
                    # set flag 'ongoing'
                    self.ongoing = True
                    # set duration
                    self.duration = 1
                    # set intensity
                    self.avg_intensity = intensity
                    # remember when alert was last raised
                    self.last_alert = time
                    # write ongoing events
                    self.write_ongoing_event()
                # or if it is a continuation of an event
                else:
                    # then we only need to update when alert was last raised
                    self.last_alert = time
                    # increment duration
                    self.duration = self.duration + 1
                    # and the average intensity
                    self.avg_intensity = update_mean(self.avg_intensity, x=intensity,
                                                     mean_smoothing=1/self.duration)
                    # Update ongoing events
                    self.update_ongoing_event()

            # when there is no alert
            else:
                # If the event is on-going
                if self.ongoing == True:
                    # Yes, the last event has not ended
                    # self.end = self.last_alert
                    # check if the last raised alert was a minute away
                    if time - self.last_alert > SAME_EVENT_TIME_BUFFER:
                        self.ongoing = False
                        # write event to file
                        self.end = self.last_alert
                        self.write_event()
                        # remove from list of ongoing events
                        self.remove_ongoing_event()
                else:  # No, the last event may not have ended
                    pass  # do nothing

        def write_event(self):
            """
            Write to file, used internally
            :return: nothing
            """
            try:
                result = self.collection_to_write_to.\
                    insert_one({
                                FIELD_NAME_EVENT_ID: self.event_id,
                                FIELD_NAME_BIN_NUMBER: self.bin_number,
                                FIELD_NAME_START: self.start,
                                FIELD_NAME_END: self.end,
                                FIELD_NAME_DURATION: self.duration,
                                FIELD_NAME_INTENSITY: round(self.avg_intensity, DIGIT_ROUNDING)
                               })
                if result.acknowledged == False:
                    logging.debug("Data not inserted:"
                                  "\n  Bin Number : %s"
                                  "\n  Start      : %s",
                                  str(self.bin_number), str(self.start))
            except Exception:
                logging.exception("Exception with data insertion:"
                                  "\n  Bin Number : %s"
                                  "\n  Start      : %s",
                                  str(self.bin_number), str(self.start))

        def write_ongoing_event(self):
            """
            Write to file, used internally
            :return: nothing
            """
            try:
                result = self.collection_for_ongoing_events.\
                    insert_one({
                    FIELD_NAME_EVENT_ID: self.event_id,
                    FIELD_NAME_BIN_NUMBER: self.bin_number,
                    FIELD_NAME_START: self.start,
                    FIELD_NAME_DURATION: self.duration,
                    FIELD_NAME_INTENSITY: round(self.avg_intensity, DIGIT_ROUNDING)
                })
                if result.acknowledged == False:
                    logging.debug("Data not inserted:"
                                  "\n  Bin Number : %s"
                                  "\n  Start      : %s",
                                  str(self.bin_number), str(self.start))
            except Exception:
                logging.exception("Exception with data insertion:"
                                  "\n  Bin Number : %s"
                                  "\n  Start      : %s",
                                  str(self.bin_number), str(self.start))

        def update_ongoing_event(self):
            """
            Update ongoing event in database, used internally
            :return: nothing
            """
            try:
                result = self.collection_for_ongoing_events.\
                    update({FIELD_NAME_EVENT_ID: self.event_id},
                           {"$set": {
                               FIELD_NAME_DURATION: self.duration,
                               FIELD_NAME_INTENSITY: round(self.avg_intensity, DIGIT_ROUNDING),
                           }
                           })
                if result['updatedExisting'] == False:
                    logging.debug("Ongoing event data not inserted:"
                                  "\n  Bin Number : %s"
                                  "\n  Start      : %s",
                                  str(self.bin_number), str(self.start))
            except Exception:
                logging.exception("Exception with ongoing event data update:"
                                  "\n  Bin Number : %s"
                                  "\n  Start      : %s",
                                  str(self.bin_number), str(self.start))

        def remove_ongoing_event(self):
            """
            remove ongoing event in database (after it has ended), used internally
            :return: nothing
            """
            try:
                result = self.collection_for_ongoing_events. \
                    remove({FIELD_NAME_EVENT_ID: self.event_id})
                if result['n'] != 1:
                    logging.debug("Multiple ongoing event data not removed:"
                                  "\n  Bin Number : %s"
                                  "\n  Start      : %s"
                                  "\n  n          : %s",
                                  str(self.bin_number), str(self.start), str(result['n']))
            except Exception:
                logging.exception("Exception with ongoing event data removal:"
                                  "\n  Bin Number : %s"
                                  "\n  Start      : %s",
                                  str(self.bin_number), str(self.start))


# *************************************************************************** #
# ====== The Diagnostics class   *This needs some Pythonification!!* ======== #
# ====== Only one instance is maintained                             ======== #
# *************************************************************************** #

class Diagnostics(object):

    __slots__ = ['configurations_collection',
                 'PCA_estimations_collection',
                 'residual_controls_collection',
                 'detection_module_instance_identifier']

    def __init__(self, configurations, PCA_estimations, residual_controls):
        """
        Initialize the Diagnostics
        :param configurations: MongoDB collection
        :param PCA_estimations: MongoDB collection
        :param residual_controls: MongoDB collection
        """
        self.configurations_collection = configurations
        self.PCA_estimations_collection = PCA_estimations
        self.residual_controls_collection = residual_controls
        self.detection_module_instance_identifier = self.configurations_collection.count()

    def write_configurations(self):
        self.configurations_collection.insert_one({
            'ID': self.detection_module_instance_identifier,
            'BRICK_DIMENSION': BRICK_DIMENSION,
            'NUMBER_OF_BINS_FOR_KNOWN_SERVERS': NUMBER_OF_BINS_FOR_KNOWN_SERVERS,
            'TIME_INTERVAL_TO_UPDATE': TIME_INTERVAL_TO_UPDATE,
            'STARTING_BATCH_SIZE': STARTING_BATCH_SIZE,
            'L_CTRL_LMT_MULTIPLE': L_CTRL_LMT_MULTIPLE,
            'ROBUST_ESTIMATION_GAURD ': ROBUST_ESTIMATION_GAURD,
            'MAX_PCA_MEMORY': MAX_PCA_MEMORY,
            'ARP_SIZE': ARP_SIZE,
            'PARAMETER_ESTIMATION_SMOOTHING': PARAMETER_ESTIMATION_SMOOTHING,
            'SAME_EVENT_TIME_BUFFER': SAME_EVENT_TIME_BUFFER,
            'MONGO_HOST': MONGO_HOST,
            'PORT': PORT,
            'DB': DB,
            'READ_COLLECTION': READ_COLLECTION,
            'WRITE_COLLECTION': WRITE_COLLECTION,
            'ONGOING_EVENT_COLLECTION': ONGOING_EVENT_COLLECTION,
            'FIELD_NAME_EVENT_ID': FIELD_NAME_EVENT_ID,
            'FIELD_NAME_BIN_NUMBER': FIELD_NAME_BIN_NUMBER,
            'FIELD_NAME_START': FIELD_NAME_START,
            'FIELD_NAME_END': FIELD_NAME_END,
            'FIELD_NAME_DURATION': FIELD_NAME_DURATION,
            'FIELD_NAME_INTENSITY': FIELD_NAME_INTENSITY,
            'DIGIT_ROUNDING': DIGIT_ROUNDING
        })

    def write_PCA_estimations(self, is_dest, timestamp, variance_explained, PCs):
        self.PCA_estimations_collection.insert_one({
            'ID': self.detection_module_instance_identifier,
            'IS_DEST': is_dest,
            'TIMESTAMP': timestamp,
            'VARIANCE_EXPLAINED': variance_explained,
            'TOP_K_PC': PCs.tolist() # cPickle may be a lot faster to read
        })

    def write_residual_controls(self, is_dest, timestamp, residuals, CL):
        self.residual_controls_collection.insert_one({
            'ID': self.detection_module_instance_identifier,
            'IS_DEST': is_dest,
            'TIMESTAMP': timestamp,
            'RESIDUALS': residuals.tolist(),
            'CONTROL_LIMIT': CL.tolist()
        })


# *************************************** #
# ===== The main driver program ========= #
# *************************************** #

if __name__ == '__main__':

    # establish connection with mongoDB
    client = MongoClient(MONGO_HOST, PORT)
    db = client[DB]
    read_collection = db[READ_COLLECTION]
    write_collection = db[WRITE_COLLECTION]
    ongoing_event_collection = db[ONGOING_EVENT_COLLECTION]
    # diagnostics database
    diagnostics_db = client[DIAGNOSTICS_DB]
    configurations_collection = diagnostics_db[CONFIGURATIONS_COLLECTION]
    PCA_estimations_collection = diagnostics_db[PCA_ESTIMATIONS_COLLECTION]
    residual_controls_collection = diagnostics_db[RESIDUAL_CONTROLS_COLLECTION]

    # Create a diagnostics tool
    diagnostics = Diagnostics(configurations=configurations_collection,
                              PCA_estimations=PCA_estimations_collection,
                              residual_controls=residual_controls_collection)
    # Record configuration parameters
    diagnostics.write_configurations()

    # Create a list of event recorders; we maintain a event recorder for each stream
    event_recorder_list = [EventRecorder(i+1, write_collection, ongoing_event_collection)
                           for i in range(NUMBER_OF_BINS_FOR_KNOWN_SERVERS, BRICK_DIMENSION)]

    # Retrieving initial batch of data for warm-up
    sum_of_src_databricks, logaritmic_sum_of_src_databricks, \
        sum_of_dst_databricks, logaritmic_sum_of_dst_databricks, \
        times = preprocessing(read_collection, STARTING_BATCH_SIZE)
    ACTUAL_STARTING_BATCH_SIZE = len(times)

    # PC estimation with initial batch of data
    stream_PCA_estimations_destination = StreamPcaEstimations(logaritmic_sum_of_dst_databricks)
    # Obtain residuals from initial batch
    initial_batch_residuals = stream_PCA_estimations_destination.initial_residuals(logaritmic_sum_of_dst_databricks)
    # use initial batch of residuals to warm up monitors. We maintain a monitor for each stream
    stream_monitor_list = [StreamMonitor(initial_batch_residuals[:, i])
                           for i in range(BRICK_DIMENSION - NUMBER_OF_BINS_FOR_KNOWN_SERVERS)]
    # record the time of the last frame of databrick
    time_of_last_frame = times[0]
    print("Warm-up Done")

    # An infinite loop that runs every TIME_INTERVAL_TO_UPDATE seconds,
    # update estimations and monitors with a new databrick
    while 1:
        # Retrieve the latest databrick
        sum_of_src_databricks, logaritmic_sum_of_src_databricks, \
            sum_of_dst_databricks, logaritmic_sum_of_dst_databricks, \
            times = preprocessing(read_collection, 1)

        # Guard against NaN values in 'logaritmic_sum_of_dst_databricks'
        if np.any(np.isnan(logaritmic_sum_of_dst_databricks)) is False:

            # Guard against repeated databricks,
            # databricks are not recorded in TIME_INTERVAL_TO_UPDATE-second-intervals sharp
            if time_of_last_frame != times[0]:

                # Update the PC estimations
                residuals = stream_PCA_estimations_destination.update(logaritmic_sum_of_dst_databricks)
                print(times[0])
                # Record PC estimations
                diagnostics.write_PCA_estimations(is_dest=True,
                                                  timestamp=times[0],
                                                  variance_explained=stream_PCA_estimations_destination.variance_explained,
                                                  PCs=stream_PCA_estimations_destination.first_k_eigenvectors)

                # Monitor each stream
                control_limits = np.array([])
                for i in range(BRICK_DIMENSION - NUMBER_OF_BINS_FOR_KNOWN_SERVERS):
                    # Update monitors
                    stream_monitor_list[i].update(x=residuals[i])
                    control_limits = np.append(control_limits,
                                               np.sqrt(stream_monitor_list[i].residual_maringal_variance) * L_CTRL_LMT_MULTIPLE)

                    # Raise alerts
                    simple_alert, simple_intensity, \
                        ewma_alert, ewma_intensity = stream_monitor_list[i].alerts(L=L_CTRL_LMT_MULTIPLE)

                    # Update and record events that have ended
                    event_recorder_list[i].update(alert=simple_alert, intensity=simple_intensity, time=times[0])

                # Record residuals and control limits
                diagnostics.write_residual_controls(is_dest=True, timestamp=times[0],
                                                    residuals=residuals, CL=control_limits)

        else:
            print(times[0], " has NAN")

        # Record the time of the last frame
        time_of_last_frame = times[0]

        # Wait for 3 seconds
        time_t.sleep(TIME_INTERVAL_TO_UPDATE)
