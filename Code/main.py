#importing required libraries
import os
import sys
from utilities import utils
from pyspark.sql import SparkSession, Window
from pyspark.sql import functions as F
from pyspark.sql.functions import col, row_number

#system file check
if os.path.exists('src.zip'):
    sys.path.insert(0, 'src.zip')
else:
    sys.path.insert(0, './Code/src')

class USVehicleAccidentAnalysis:
    def __init__(self, path_to_config_file):
        input_file_paths = utils.read_yaml(path_to_config_file).get("INPUT_FILENAME")
        self.df_charges = utils.load_csv_data_to_df(spark, input_file_paths.get("Charges"))
        self.df_damages = utils.load_csv_data_to_df(spark, input_file_paths.get("Damages"))
        self.df_endorse = utils.load_csv_data_to_df(spark, input_file_paths.get("Endorse"))
        self.df_primary_person = utils.load_csv_data_to_df(spark, input_file_paths.get("Primary_Person"))
        self.df_units = utils.load_csv_data_to_df(spark, input_file_paths.get("Units"))
        self.df_restrict = utils.load_csv_data_to_df(spark, input_file_paths.get("Restrict"))
        spark.conf.set("spark.sql.repl.eagerEval.enabled", False)

    #Analytics_1
    def crashes_with_more_than_2_males_killed(self, output_path, output_format):
        """
        Find the number of crashes (accidents) in which number of males killed are greater than 2
        """
        male_fatalities = self.df_primary_person.filter((F.col("PRSN_GNDR_ID") == "MALE") & (F.col("PRSN_INJRY_SEV_ID") == "KILLED"))
        male_fatalities_crash_count = male_fatalities.groupBy("CRASH_ID").agg(F.count("CRASH_ID").alias("male_fatalities_count"))
        crashes_with_more_than_2_males_killed = male_fatalities_crash_count.filter(F.col("male_fatalities_count") > 2)
        utils.write_output(crashes_with_more_than_2_males_killed, output_path, output_format)
        return crashes_with_more_than_2_males_killed.count()

    #Analytics_2
    def two_wheelers_booked_for_crash(self, output_path, output_format):
        """
        How many two wheelers are booked for crashes
        """
        two_wheelers = self.df_units.filter(F.col("VEH_BODY_STYL_ID").contains("MOTORCYCLE"))
        utils.write_output(two_wheelers, output_path, output_format)
        return two_wheelers.count()

    #Analytics_3
    def top_5_makes(self, output_path, output_format):
        """
        Determine the Top 5 Vehicle Makes of the cars present in the crashes in which driver died and Airbags did not deploy.
        """
        filtered_data_3 = self.df_primary_person.filter((F.col("DEATH_CNT") > 0) & (F.col("PRSN_AIRBAG_ID") == "NOT DEPLOYED"))
        joined_data_3 = filtered_data_3.join(self.df_units, "CRASH_ID", "inner")
        make_counts = joined_data_3.groupBy("VEH_MAKE_ID").count().orderBy(F.desc("count"))
        top_5_makes = make_counts.filter(make_counts.VEH_MAKE_ID != "NA").limit(5).select("VEH_MAKE_ID")
        utils.write_output(top_5_makes, output_path, output_format)
        return top_5_makes.show()

    #Analytics_4
    def driver_with_valid_licence(self, output_path, output_format):
        """
        Determine number of Vehicles with driver having valid licences involved in hit and run
        """
        hit_and_run_df = self.df_charges.filter((F.col("CHARGE") == "HIT AND RUN") & (F.col("UNIT_NBR").isNotNull()))
        joined_df_4 = hit_and_run_df.join(self.df_pp, on=['CRASH_ID', 'UNIT_NBR'], how='inner')
        valid_license_df = joined_df_4.filter(F.col("DRVR_LIC_TYPE_ID").isNotNull())
        result_count = valid_license_df.select('CRASH_ID', 'UNIT_NBR').distinct().count()
        utils.write_output(result_count, output_path, output_format)
        return result_count

    #Analytics_5
    def max_accident_state(self, output_path, output_format):
        """
        Which state has highest number of accidents in which females are not involved
        """
        joined_df_5 = self.df_primary_person.join(self.df_units, on=['CRASH_ID', 'UNIT_NBR'], how='inner')
        accidents_without_females = joined_df_5.filter(F.col("PRSN_GNDR_ID") != 'Female')
        state_accident_counts = accidents_without_females.groupBy("VEH_LIC_STATE_ID").count()
        max_accident_state = state_accident_counts.orderBy(F.desc("count")).first()["VEH_LIC_STATE_ID"]
        utils.write_output(max_accident_state, output_path, output_format)
        return max_accident_state

    #Analytics_6
    def veh_make_id(self, output_path, output_format):
        """
        Which are the Top 3rd to 5th VEH_MAKE_IDs that contribute to a largest number of injuries including death
        """
        injury_counts = self.df_units.groupBy("VEH_MAKE_ID").agg(
            F.sum("TOT_INJRY_CNT").alias("TOTAL_INJURY_COUNT"),
            F.sum("DEATH_CNT").alias("DEATH_COUNT")
        )
        injury_counts = injury_counts.withColumn(
            "TOTAL_INJURY_COUNT_INCLUDING_DEATH",
            F.col("TOTAL_INJURY_COUNT") + F.col("DEATH_COUNT")
        )
        window_spec = Window.orderBy(F.desc("TOTAL_INJURY_COUNT_INCLUDING_DEATH"))
        injury_ranking = injury_counts.withColumn(
            "rank",
            F.rank().over(window_spec)
        )
        result = injury_ranking.filter((F.col("rank") >= 3) & (F.col("rank") <= 5)).select("VEH_MAKE_ID")
        utils.write_output(result, output_path, output_format)
        return result.show()

    #Analytics_7
    def top_ethnic_user_group(self, output_path, output_format):
        """
        For all the body styles involved in crashes, mention the top ethnic user group of each unique body style
        """
        df_pp_selected = self.df_primary_person.select("CRASH_ID", "PRSN_ETHNICITY_ID")
        df_Units_selected = self.df_units.select("CRASH_ID", "VEH_BODY_STYL_ID")
        joined_df_7 = df_Units_selected.join(df_pp_selected, "CRASH_ID", "inner")
        filtered_joined_df_7 = joined_df_7.filter(
            ~joined_df_7.VEH_BODY_STYL_ID.isin("NA", "NOT REPORTED", "OTHER", "UNKNOWN")
        )
        grouped_df_7 = filtered_joined_df_7.groupBy("VEH_BODY_STYL_ID", "PRSN_ETHNICITY_ID").agg(
            F.count("PRSN_ETHNICITY_ID").alias("ethnicity_count")
        )
        window_spec = Window.partitionBy("VEH_BODY_STYL_ID").orderBy(F.desc("ethnicity_count"))
        ranked_df = grouped_df_7.withColumn(
            "rank",
            F.rank().over(window_spec)
        )
        result = ranked_df.filter(F.col("rank") == 1).select("VEH_BODY_STYL_ID", "PRSN_ETHNICITY_ID")
        utils.write_output(result, output_path, output_format)
        return result.show(truncate=False)

    #Analytics_8
    def top_zip_codes(self, output_path, output_format):
        """
        Among the crashed cars, what are the Top 5 Zip Codes with highest number crashes with alcohols as the contributing factor to a crash (Use Driver Zip Code)
        """
        df_selected_8 = self.df_primary_person.select("CRASH_ID", "DRVR_ZIP", "PRSN_ALC_RSLT_ID")
        df_filtered_8 = df_selected_8.filter(F.col("PRSN_ALC_RSLT_ID") == "Positive")
        df_filtered_8 = df_filtered_8.filter(df_filtered_8.DRVR_ZIP.isNotNull())
        zip_code_counts = df_filtered_8.groupBy("DRVR_ZIP").agg(
            F.count("CRASH_ID").alias("crash_count")
        )
        top_zip_codes = zip_code_counts.sort(F.desc("crash_count")).limit(5).select("DRVR_ZIP")
        utils.write_output(top_zip_codes, output_path, output_format)
        return top_zip_codes.show(truncate=False)

    #Analytics_9
    def distinct_crash_count(self, output_path, output_format):
        """
        Count of Distinct Crash IDs where No Damaged Property was observed and Damage Level (VEH_DMAG_SCL~) is above 4 and car avails Insurance
        """
        df_damages_selected = self.df_damages.select("CRASH_ID", "DAMAGED_PROPERTY")
        df_units_selected = self.df_units.select("CRASH_ID", "VEH_DMAG_SCL_1_ID", "FIN_RESP_PROOF_ID")
        joined_df_9 = df_damages_selected.join(df_units_selected, "CRASH_ID", "inner")
        filtered_df_9 = joined_df_9.filter(
            (F.col("DAMAGED_PROPERTY").isNull()) &
            (F.col("VEH_DMAG_SCL_1_ID").isin('DAMAGED 5', 'DAMAGED 7 HIGHEST', 'DAMAGED 6')) &
            (F.col("FIN_RESP_PROOF_ID").like('%INSURANCE%'))
        )
        distinct_crash_count = filtered_df_9.select("CRASH_ID").distinct().count()
        utils.write_output(distinct_crash_count, output_path, output_format)
        return distinct_crash_count

    #Analytics_10
    def top_vehicle_makes(self, output_path, output_format):
        """
        Determine the Top 5 Vehicle Makes where drivers are charged with speeding related offences, has licensed Drivers, 
        used top 10 used vehicle colours and has car licensed with the Top 25 states with highest number of offences (to be deduced from the data)
        """
        df_charges_filtered_10 = self.df_charges.select("CRASH_ID", "UNIT_NBR", "PRSN_NBR", "CHARGE")
        df_pp_filtered_10 = self.df_primary_person.select("CRASH_ID", "UNIT_NBR", "DRVR_LIC_STATE_ID")
        df_Units_filtered_10 = self.df_units.select("CRASH_ID", "VEH_MAKE_ID", "VEH_COLOR_ID", "VEH_LIC_STATE_ID")
        top_colors = ["WHI", "BLK", "SIL", "GRY", "BLU", "RED", "GRN", "MAR", "TAN", "GLD"]
        top_states = ["Texas", "Mexico", "Louisiana", "New Mexico", "California", "Florida", "Oklahoma", "Arkansas",
                      "Arizona", "Georgia", "Colorado", "Illinois", "Tennessee", "Missouri", "Mississippi",
                      "North Carolina", "Kansas", "Alabama", "Michigan", "Washington", "Ohio", "New York", "Virginia",
                      "South Carolina", "Pennsylvania"]
        filtered_data_10 = df_charges_filtered_10.join(df_pp_filtered_10, on="CRASH_ID", how="inner") \
            .join(df_Units_filtered_10, on="CRASH_ID", how="inner") \
            .filter((F.col("CHARGE").contains("SPEED")) &
                    (F.col("DRVR_LIC_STATE_ID").isin(top_states)) &
                    (F.col("VEH_COLOR_ID").isin(top_colors)))
        top_vehicle_makes = filtered_data_10.groupBy("VEH_MAKE_ID").count().orderBy(F.desc("count")).limit(5) \
            .select("VEH_MAKE_ID")
        utils.write_output(top_vehicle_makes, output_path, output_format)
        return top_vehicle_makes.show(truncate=False)

if __name__ == '__main__':
    # Initialize sparks session
    spark = SparkSession \
        .builder \
        .appName("USVehicleAccidentAnalysis") \
        .getOrCreate()

    config_file_path = "config.yaml"
    spark.sparkContext.setLogLevel("ERROR")

    us_vehicle_accident_analysis = USVehicleAccidentAnalysis(config_file_path)

    # Output paths and file formats
    output_file_paths = utils.read_yaml(config_file_path).get("OUTPUT_PATH")
    file_format = utils.read_yaml(config_file_path).get("FILE_FORMAT")

    #Find the number of crashes (accidents) in which number of males killed are greater than 2?
    result_1 = us_vehicle_accident_analysis.crashes_with_more_than_2_males_killed(output_file_paths.get("Analytics_1"), file_format.get("Output"))
    print("Analytics_1 Result:", result_1)

    #How many two wheelers are booked for crashes?
    result_2 = us_vehicle_accident_analysis.two_wheelers_booked_for_crash(output_file_paths.get("Analytics_2"), file_format.get("Output"))
    print("Analytics_2 Result:", result_2)

    #Determine the Top 5 Vehicle Makes of the cars present in the crashes in which driver died and Airbags did not deploy.
    result_3 = us_vehicle_accident_analysis.top_5_makes(output_file_paths.get("Analytics_3"), file_format.get("Output"))
    print("Analytics_3 Result:", result_3)

    #Determine number of Vehicles with driver having valid licences involved in hit and run?
    result_4 = us_vehicle_accident_analysis.driver_with_valid_licence(output_file_paths.get("Analytics_4"), file_format.get("Output"))
    print("Analytics_4 Result:", result_4)

    #Which state has highest number of accidents in which females are not involved?
    result_5 = us_vehicle_accident_analysis.max_accident_state(output_file_paths.get("Analytics_5"), file_format.get("Output"))
    print("Analytics_5 Result:", result_5)

    #Which are the Top 3rd to 5th VEH_MAKE_IDs that contribute to a largest number of injuries including death
    result_6 = us_vehicle_accident_analysis.veh_make_id(output_file_paths.get("Analytics_6"), file_format.get("Output"))
    print("Analytics_6 Result:", result_6)

    #For all the body styles involved in crashes, mention the top ethnic user group of each unique body style
    result_7 = us_vehicle_accident_analysis.top_ethnic_user_group(output_file_paths.get("Analytics_7"), file_format.get("Output"))
    print("Analytics_7 Result:", result_7)

    #Among the crashed cars, what are the Top 5 Zip Codes with highest number crashes with alcohols as the contributing factor to a crash (Use Driver Zip Code)
    result_8 = us_vehicle_accident_analysis.top_zip_codes(output_file_paths.get("Analytics_8"), file_format.get("Output"))
    print("Analytics_8 Result:", result_8)

    #Count of Distinct Crash IDs where No Damaged Property was observed and Damage Level (VEH_DMAG_SCL~) is above 4 and car avails Insurance
    result_9 = us_vehicle_accident_analysis.distinct_crash_count(output_file_paths.get("Analytics_9"), file_format.get("Output"))
    print("Analytics_9 Result:", result_9)

    #Determine the Top 5 Vehicle Makes where drivers are charged with speeding related offences, has licensed Drivers, used top 10 used vehicle colours and has car licensed with the Top 25 states with highest number of offences (to be deduced from the data)
    result_10 = us_vehicle_accident_analysis.top_vehicle_makes(output_file_paths.get("Analytics_10"), file_format.get("Output"))
    print("Analytics_10 Result:", result_10)


# Stop the Spark session
spark.stop()
