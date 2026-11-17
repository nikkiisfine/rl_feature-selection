from pathlib import Path
import pandas as pd
import duckdb

# ========= 1) 配置输入输出路径 =========
BASE_DIR = Path(__file__).resolve().parent  # 脚本所在目录

XLSX_PATH_CLINICAL = BASE_DIR / "All_Subjects_My_Table_27Jan2026.xlsx"
XLSX_PATH = BASE_DIR / "T1 cohort manifest.xlsx"

# 输出文件
OUT_CSV = BASE_DIR / "joined_output.csv"
# OUT_XLSX = BASE_DIR / "joined_output.xlsx"

# ========= 2) 读取数据 =========
def load_inputs(clinical_path: Path, xlsx_path: Path):
    if not clinical_path.exists():
        raise FileNotFoundError(f"CSV not found: {clinical_path}")
    if not xlsx_path.exists():
        raise FileNotFoundError(f"XLSX not found: {xlsx_path}")

    df_all = pd.read_excel(clinical_path, sheet_name = 'All_Subjects_DXSUM_27Jan2026')

    # 如果 xlsx 不是第一个 sheet，改 sheet_name，比如 sheet_name="Sheet1"
    df_t1 = pd.read_excel(xlsx_path, sheet_name='Sheet4')  #t1
    df_t2 = pd.read_excel(xlsx_path, sheet_name ='Clinical_fmri_Imaging_Cohort')   #t2

    return df_all, df_t1,df_t2

SQL_QUERY = r"""
select t.PID1 id, t.cli_date_1 clinical_date,
        case when t.diag_1 =1 then 0 
            when t.diag_1 =3  then 1 
        end as AD_label,
        t.T1w_image_id, t.T1w,t.T1w_date,
        t.rs_fmri_image_id,t.rs_fmri,t.rs_fmri_date
from
    (SELECT t0.PID1, t0.cli_date_1, t0.diag_1, 
            t0.T1w_image_id, t0.T1w_date,t0.T1w,
            t0.rs_fmri_Image_ID, t0.rs_fmri_Date, t0.rs_fmri,
            datediff('day',CAST(t0.cli_date_1 AS DATE), CAST(t0.T1w_date AS DATE)) T1w_in_180,
            datediff('day',CAST(t0.cli_date_1 AS DATE), CAST(t0.rs_fmri_date AS DATE)) rs_fmri_in_180
    FROM 
        (select t3.*, t4.* 
            from
                (select t1.subject_id, t1.T1w_image_id, t1.T1w_date,t1.T1w, a.PTID PID1,a.EXAMDATE cli_date_1,a.DIAGNOSIS diag_1
                    from all_subjects_clinical a
                    left join t1_T1w t1
                    on a.PTID = t1.subject_id
                    where t1.subject_id is not null and a.DIAGNOSIS in (1,3) 
                ) t3
            join
                (select t2.Subject_ID,t2.rs_fmri_Image_ID, t2.rs_fmri_Date, t2.rs_fmri, a.PTID PID2,a.EXAMDATE cli_date_2,a.DIAGNOSIS diag_2
                    from all_subjects_clinical a
                    left join t2_rs_fmri t2
                    on a.PTID = t2.Subject_ID
                    where t2.Subject_ID is not null and a.DIAGNOSIS in (1,3)
                    and 
                    (
                        (LOWER(t2.rs_fmri) LIKE '%rsfmri%'
                        OR LOWER(t2.rs_fmri) LIKE '%rs fmri%'
                        OR LOWER(t2.rs_fmri) LIKE '%resting state fMRI%'
                        )
                    AND LOWER(t2.rs_fmri) NOT LIKE '%asl%'
                    AND LOWER(t2.rs_fmri) NOT LIKE '%pasl%'
                    AND LOWER(t2.rs_fmri) NOT LIKE '%pcasl%'
                    AND LOWER(t2.rs_fmri) NOT LIKE '%perfusion%'
                    AND LOWER(t2.rs_fmri) NOT LIKE '%cbf%'
                    AND LOWER(t2.rs_fmri) NOT LIKE '%dti%'
                    AND LOWER(t2.rs_fmri) NOT LIKE '%moco%'
                    AND LOWER(t2.rs_fmri) NOT LIKE '%repeat%'
                    AND LOWER(t2.rs_fmri) NOT LIKE '%extended%'
                    AND LOWER(t2.rs_fmri) NOT LIKE '%AP%'
                    AND LOWER(t2.rs_fmri) NOT LIKE '%mb%'
                    )
                )t4
            on t3.PID1 = t4.PID2 and t3.cli_date_1 = t4.cli_date_2 and t3.diag_1 = t4.diag_2
        )t0
    )t  
    where abs(t.T1w_in_180) <= 180 and abs(t.rs_fmri_in_180) <= 180
;
"""

def run_sql(df_all: pd.DataFrame, df_t1: pd.DataFrame, df_t2: pd.DataFrame, sql: str) -> pd.DataFrame:
    con = duckdb.connect(database=":memory:")

    # 注册 pandas DataFrame 为 SQL 表
    con.register("all_subjects_clinical", df_all)
    con.register("t1_T1w", df_t1)
    con.register('t2_rs_fmri', df_t2)

    try:
        out_df = con.execute(sql).fetchdf()
    finally:
        con.close()

    return out_df

# ========= 4) 导出 =========
def export_outputs(df: pd.DataFrame, out_csv: Path):
    df.to_csv(out_csv, index=False)
    # df.to_excel(out_xlsx, index=False)
    print(f"[OK] Saved CSV : {out_csv}")
    # print(f"[OK] Saved XLSX: {out_xlsx}")
    print(f"[OK] Rows: {len(df):,} | Cols: {df.shape[1]:,}")


def main():
    df_all, df_t1,df_t2 = load_inputs(XLSX_PATH_CLINICAL, XLSX_PATH)

    print("XLSX_CLINICAL columns:", list(df_all.columns))
    print("XLSX_T1w_SCAN columns:", list(df_t1.columns))
    print("XLSX_rs_fmri_SCAN columns:", list(df_t2.columns))

    out_df = run_sql(df_all, df_t1,df_t2, SQL_QUERY)
    export_outputs(out_df, OUT_CSV)


if __name__ == "__main__":
    main()
