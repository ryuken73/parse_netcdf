import netCDF4 as nc
import numpy as np
import json

def convert_to_lonlat(x, y, projection_info):
    """GK-2A 정지궤도(GEOS) 투영을 경위도로 변환 (개발 매뉴얼 기반, 개선됨)"""
    # 상수 정의
    degtrrad = 0.017453292519943295  # 라디안 변환 상수

    # 투영 파라미터
    COFF = projection_info.column_offset  # 2750.5
    LOFF = projection_info.line_offset    # 2750.5
    CFAC = projection_info.column_scale_factor  # 2.0425338E7
    LFAC = projection_info.line_scale_factor    # 2.0425338E7
    sub_lon = projection_info.longitude_of_projection_origin  # 128.2°

    # 픽셀 좌표를 라디안으로 변환
    x = (x - COFF) * degtrrad * 16 / CFAC  # 열 좌표 변환
    y = (LOFF - y) * degtrrad * 16 / LFAC  # 행 좌표 변환 (y축 반전)

    # 중간 계산
    sd = np.sqrt(x**2 + y**2)
    
    # sd == 0 처리 (특이점 처리)
    mask = sd == 0
    sn = np.zeros_like(sd, dtype=float)
    sn[~mask] = (np.cos(x[~mask]) * np.cos(y[~mask]) - 1 / sd[~mask]) / (
        np.sin(x[~mask])**2 * np.cos(y[~mask])**2 + np.sin(y[~mask])**2 + 1.006739501 * np.sin(y[~mask])**2
    )
    sn[mask] = -1  # 특이점(0,0)에서는 sn = -1로 설정

    # s1, s2, s3 계산 (sd == 0 방지)
    s1 = np.sin(x) * np.cos(y) / np.where(sd != 0, sd, 1)
    s2 = -np.sin(y) / np.where(sd != 0, sd, 1)
    # s3 계산 재조정 (GK-2A 특성 반영)
    s3 = (np.sin(x) * np.cos(y) * sn / np.where(sd != 0, sd, 1) - np.cos(x) * np.cos(y)) * 1.006739501

    # 디버깅: s3 값 범위 확인
    print(f"s3 range: {np.min(s3):.2f} ~ {np.max(s3):.2f}")

    # 위도 계산
    lat = np.arcsin(s3) / degtrrad

    # 경도 계산
    lon = sub_lon + np.arctan2(s1, s2) / degtrrad

    # 경도 정규화 (-180 ~ 180)
    lon = np.where(lon > 180, lon - 360, lon)
    lon = np.where(lon < -180, lon + 360, lon)

    return lon, lat

def extract_ctt_data(nc_file_path, step=1):
    ds = nc.Dataset(nc_file_path, 'r')
    
    # CTT 데이터 읽기 및 검증
    try:
        ctt_raw = ds.variables['CTT'][:]
        print(f"CTT raw sample: {ctt_raw[0,0]}")  # CTT 데이터 샘플 출력
    except Exception as e:
        print(f"Error reading CTT data: {e}")
    
    ctt_fill = ds.variables['CTT']._FillValue
    scale_factor = ds.variables['CTT'].scale_factor
    add_offset = ds.variables['CTT'].add_offset
    
    # 마스크 처리 후 스케일링
    ctt = np.ma.masked_where(ctt_raw == ctt_fill, ctt_raw)
    ctt = ctt * scale_factor + add_offset  # K 단위로 변환
    print(f"CTT scaled sample: {ctt[0,0]}")  # 스케일링 후 샘플 출력

    proj = ds.variables['gk2a_imager_projection']
    ydim, xdim = ctt.shape
    x = np.arange(xdim)
    y = np.arange(ydim)
    xx, yy = np.meshgrid(x, y)
    
    lon, lat = convert_to_lonlat(xx, yy, proj)
    
    # 범위 확인
    print(f"Lon range: {np.min(lon):.2f} ~ {np.max(lon):.2f}")
    print(f"Lat range: {np.min(lat):.2f} ~ {np.max(lat):.2f}")
    
    result = []
    for i in range(0, ydim, step):
        for j in range(0, xdim, step):
            if not np.ma.is_masked(ctt[i, j]):
                result.append([
                    float(lon[i, j]),  # lon
                    float(lat[i, j]),  # lat
                    float(ctt[i, j])   # ctt_value
                ])
    
    output_file = f"output_fd_{step}K.json"
    with open(output_file, 'w') as f:
        json.dump(result, f)
    
    ds.close()
    return result

if __name__ == "__main__":
    file_path = r"D:\002.Code\001.python\netcdf\ctps_fd_ge_202502170000.nc"
    step_size = 10
    
    data = extract_ctt_data(file_path, step=step_size)
    print(f"Data saved to output_fd_{step_size}K.json")
    print(f"First 5 records: {data[:5]}")