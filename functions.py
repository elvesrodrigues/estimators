import json 
import random
import datetime
import math
import matplotlib.pyplot as plt

import numpy as np
from scipy.optimize import fsolve

class DatetimeHandler:
    def __init__(self):
        with open('calendar.json') as f:
            self.calendar = json.loads(f.read())
            f.close()

    def split_timestamp(self, timestamp):
        timestamp = str(timestamp)
        return int(timestamp[0:4]), int(timestamp[4:6]), int(timestamp[6:8]), int(timestamp[8:10]), int(timestamp[10:12]), int(timestamp[12:14]) 

    def timestamp_plus_offset(self, timestamp, days_offset = 0, hours_offset = 0, minutes_offset = 0, seconds_offset = 0):
        last_year, last_month, last_day, last_hour, last_mins, last_secs = self.split_timestamp(timestamp)
        
        curr_secs = (last_secs + seconds_offset) % 60
        minutes_offset += (last_secs + seconds_offset) // 60
        
        curr_mins = (last_mins + minutes_offset) % 60
        hours_offset += (last_mins + minutes_offset) // 60
        
        curr_hour = (last_hour + hours_offset) % 24
        days_offset += (last_hour + hours_offset) // 24
        
        last_day_in_month = int(self.calendar[f'{last_year:02}'][f'{last_month:02}'][-1])

        curr_day = max((last_day + days_offset) % (last_day_in_month + 1), 1)
        month_offset = (last_day + days_offset) // (last_day_in_month + 1)
            
        curr_month = max((last_month + month_offset) % 13, 1)
        year_offset = (last_month + month_offset) // 13
        
        curr_year = last_year + year_offset
        
        return f'{curr_year}{curr_month:02}{curr_day:02}{curr_hour:02}{curr_mins:02}{curr_secs:02}'

    def timestamp_to_datetime(self, timestamp):
        year, month, day, hour, minute, secs = self.split_timestamp(timestamp)
        return f'{year}/{month:02}/{day:02} {hour:02}:{minute:02}:{secs:02}'

class Simulator:
    def __init__(self):
        self.datetime_handler = DatetimeHandler()

        with open('site-timestamps.json') as f:
            self.historic = json.loads(f.read())
            f.close() 

        with open('date-to-offset.json') as f:
            self.datetime_to_offset = json.loads(f.read())
            f.close()

        self.parameters = None
        self.intercept = None
    
    def calculate_changes_by_offset(self, site_name):
        offset_changes = dict([(offset, 0) for offset in range(len(self.datetime_to_offset))])

        for timestamp in self.historic[site_name]:
            datetime = str(timestamp)[:8]

            offset_changes[self.datetime_to_offset[datetime]] += 1

        offsets = list()
        changes = list()

        for offset in offset_changes:
            offsets.append(offset)
            changes.append(offset_changes[offset])

        return (site_name, offsets, changes)

    def generate_regular_timestamps(self, num_intervals, init_timestamp=20191101235500, days_offset = 0, hours_offset = 24, minutes_offset = 0, seconds_offset = 0):
        timestamp = init_timestamp

        timestamps = [timestamp]
        for i in range(num_intervals-1):
            timestamp = int(self.datetime_handler.timestamp_plus_offset(timestamp, days_offset, hours_offset, minutes_offset, seconds_offset))
            timestamps.append(timestamp)
            
            # Os dados foram coletados até essa data         
            if timestamp > 20200501000000:
                break
        
        return timestamps

    # aceita apenas horas
    def generate_irregular_timestamps(self, num_intervals,
                                        init_timestamp=20191101235500, 
                                        min_hours_offset=12, 
                                        max_hours_offset=60,
                                        min_minutes_offset=0,
                                        max_minutes_offset=60,
                                        min_seconds_offset=0,
                                        max_seconds_offset=3600):
        
        timestamp = init_timestamp

        timestamps = list()
        offsets = list()

        for i in range(num_intervals):
            hours_offset = random.randint(min_hours_offset, max_hours_offset)
            minutes_offset = random.randint(min_minutes_offset, max_minutes_offset)
            seconds_offset = random.randint(min_seconds_offset, max_seconds_offset)

            timestamp = int(self.datetime_handler.timestamp_plus_offset(timestamp, hours_offset=hours_offset, minutes_offset=minutes_offset, seconds_offset=seconds_offset))
            timestamps.append(timestamp)


            offset = hours_offset * 3600 + minutes_offset * 60 + seconds_offset
            offsets.append(round(offset / 86400, 2))

            # Os dados foram coletados até essa data         
            if timestamp > 20200501000000:
                break
                
        return timestamps, offsets

    def check_changes(self, historic, ref_timestamp):
        num_changes = 0

        for timestamp in historic:
            if timestamp > ref_timestamp:
                break

            num_changes += 1
        
        return num_changes

    def count_visits_with_changes(self, historic, visits):
        changes = 0

        for visit in visits:
            num_changes = self.check_changes(historic, visit)
            historic = historic[num_changes:]

            if num_changes:
                changes += 1

        return changes

    def regular_estimator_1(self, historic, visits):
        return self.count_visits_with_changes(historic, visits) / len(visits)

    def regular_estimator_2(self, historic, visits):
        num_visits = len(visits)
        num_changes = self.count_visits_with_changes(historic, visits)

        return - math.log(((num_visits - num_changes) + .5) / (num_visits + .5))

    def generate_change_or_not_list(self, visits, offsets, historic): 
        changes = list()
        no_changes = list()

        for visit, offset in zip(visits, offsets):
            num_changes = self.check_changes(historic, visit)
            historic = historic[num_changes:]

            if num_changes:
                changes.append(offset)
            else:
                no_changes.append(offset)

        return changes, no_changes

    def f(self, x):
        expr = self.intercept

        for parameter in self.parameters:
            expr += parameter / (np.exp(parameter * x) - 1)

        return expr

    def irregular_estimator(self, historic, visits, offsets):
        changes, no_changes = self.generate_change_or_not_list(visits, offsets, historic) 

        self.parameters = changes
        self.intercept = -sum(no_changes)

        _lambda = fsolve(self.f, 1.0)[0]
        return 1 / _lambda 

    def simulate(self, site, visits, time_interval):
        historic = self.historic[site].copy()

        try:
            r1 = self.regular_estimator_1(historic, visits)
            lambda1 = r1 / time_interval

            # tempo estimado de atualização em horas
            update_freq1 = int(1 // lambda1 )

            # calcula quantos intervalos de tempos são possíveis para a frequência estimada 
            # considerando o histórico disponível
            num_intervals = int(len(self.datetime_to_offset) * 24 // update_freq1)

            visits_to_aval1 = self.generate_regular_timestamps(num_intervals, hours_offset=update_freq1)
            percent_of_visits_with_changes_1 = round(self.count_visits_with_changes(historic, visits_to_aval1) / len(visits_to_aval1), 2) * 100

        except: # A visitas feitas não foram suficientes para gerar uma estimativa
            percent_of_visits_with_changes_1 = 0
            update_freq1 = 0

        try:
            r2 = self.regular_estimator_2(historic, visits)
            lambda2 = r2 / time_interval
            # tempo estimado de atualização em horas
            update_freq2 = int(1 // lambda2)

            # calcula quantos intervalos de tempos são possíveis para a frequência estimada 
            # considerando o histórico disponível
            num_intervals = int(len(self.datetime_to_offset) * 24 // update_freq2)
            visits_to_aval2 = self.generate_regular_timestamps(num_intervals, hours_offset=update_freq2)
            percent_of_visits_with_changes_2 = round(self.count_visits_with_changes(historic, visits_to_aval2) / len(visits_to_aval2), 2) * 100

        except: # A visitas feitas não foram suficientes para gerar uma estimativa
            percent_of_visits_with_changes_2 = 0
            update_freq2 = 0

        return percent_of_visits_with_changes_1, percent_of_visits_with_changes_2, update_freq1, update_freq2    

    def run_experiment(self, site, max_intervals, hours_offset = 24):
        results = list()

        for interval in range(1, max_intervals + 1): 
            visits = self.generate_regular_timestamps(interval, hours_offset=24)
            p1, p2, u1, u2 = self.simulate(site, visits, hours_offset)

            results.append((interval, p1, p2, u1, u2))

        return results

    def plot_compare(hists, results, main_title, nrows, ncols, figx=20, figy=30):
        fig, axs = plt.subplots(nrows, ncols, figsize=(figx,figy))
        
        for ax, hist, data in zip(axs, hists, results):
            title, days, freq = hist
        
            name, content = data
            
            xs = list()
            
            y1s = list()
            y2s = list()
            
            z1s = list()
            z2s = list()
            
            for values in content:
                xs.append(values[0])
                y1s.append(values[1])
                y2s.append(values[2])
                z1s.append(values[3])
                z2s.append(values[4])
            
            ax[0].bar(days, freq)
            ax[0].set(xlabel='#dias após 31/10/2019', ylabel='# de visitas com alterações', title=f'{name}: registro de alterações')
            
            ax[1].plot(xs, y1s, '--r', label='r = X / n')
            ax[1].plot(xs, y2s, '--k', label='r = -log((n - X + .5) / (n + .5)')
            ax[1].legend()
            ax[1].set(xlabel='# dias de revisitas', ylabel='% de visitas com alterações', title=f'{name}: comparação de % de visitas com alterações')

            ax[2].plot(xs, z1s, '--r', label='r = X / n')
            ax[2].plot(xs, z2s, '--k', label='r = -log((n - X + .5) / (n + .5)')
            ax[2].legend()
            ax[2].set(xlabel='# dias de revisitas', ylabel='Tempo previsto de revisita (em horas)', title=f'{name}: comparação de tempo previsto por estimador')
            
        fig.suptitle(main_title) # or plt.suptitle('Main title')
        fig.subplots_adjust(hspace=0.4)
        fig.subplots_adjust(wspace=0.2)
        
        plt.show()
        
# if __name__ == "__main__":
#     simulator = Simulator()

#     # print(simulator.run_experiment('usp', 10))
#     visits = simulator.generate_regular_timestamps(1)
#     print(simulator.simulate('usp', visits, 24))

    # print(len(simulator.datetime_to_offset))

#     visits, offsets = simulator.generate_irregular_timestamps(10)
#     historic = simulator.historic['mg']
    
#     changes, no_changes = simulator.generate_change_or_not_list(visits, offsets, historic) 

#     simulator.intercept = -sum(no_changes)
#     simulator.parameters = changes

#     _lambda = fsolve(simulator.f, 1.0)[0]
#     # time_elapsed = sum(changes) + sum(no_changes)

#     # changes_by_day = _lambda * time_elapsed
#     change_frequency = int((1 / _lambda) * 86400)

#     # hours = change_frequency / 3600
#     # mean = (sum(offsets) / len(offsets) * 8600) / 3600

#     print(change_frequency, datetime.timedelta(seconds=change_frequency))

    # print(visits)
    # lambda_1 = simulator.regular_estimator_1(simulator.historic['g1'], visits) / 24 
    # lambda_2 = simulator.regular_estimator_2(simulator.historic['g1'], visits) / 24 
    # print(1 // lambda_1)
    # print(1 // lambda_2)
    # print(simulator.datetime_handler.timestamp_plus_offset(20191101235500, hours_offset=2, minutes_offset=24, seconds_offset=48))