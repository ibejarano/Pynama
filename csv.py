import csv
        
with open ("prueba.csv","a") as f:#save csv Error8 and time
        file_csv=csv.writer(f)
        file_csv.writerows([[1,2,3],[4,5,6]]])