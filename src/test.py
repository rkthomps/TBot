from W_Preproc import Weekly_Preprocessor as WP
import datetime
start = datetime.datetime.now()
wp = WP(40, 2010, 2020)
build_time = datetime.datetime.now()
wp.get_next_week()
end = datetime.datetime.now()

print('Build_Time', (build_time - start).total_seconds())
print('End_Time', (end - build_time).total_seconds())


