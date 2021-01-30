import click
from click_help_colors import HelpColorsGroup, HelpColorsCommand

@click.group(
    cls=HelpColorsGroup,
    help_headers_color='yellow',
    help_options_color='green'
)

def eo2cube():
    pass

@eo2cube.command()
@click.option("--db_name", default=None, help="Database name", required=True)
@click.option("--db_user", default=None, help="Database username", required=True)
@click.option("--db_pass", default=None, help="Database password", required=True)
@click.option("--host", default="127.0.01", help="Host IP")
@click.option("--port", default="5432", help="Port")
@click.option("--schema", default="agdc", help="Database schema")

def dc_products(db_name, db_user,db_pass,host, port, schema):
    """Show data cube products"""

    import psycopg2
    import pandas as pd

    conn = psycopg2.connect(user=db_user, password=db_pass, host=host, port=port, database=db_name, options=f'-c search_path={schema}')

    sql_command = "SELECT count(*), t.name FROM dataset  LEFT JOIN dataset_type t ON dataset.dataset_type_ref = t.id GROUP BY t.name;"
    datasets = pd.read_sql(sql_command, conn)
    print(datasets)
    conn.close()

    click.echo("eo2cube get_products")


@eo2cube.command()
@click.option("--aoi", default=None, help="Shapefile defining the area of interest", required=True)
@click.option("--product", default=['landsat8'], help="Define which data to search for (either landsat5,landsat7 or landsat8)", required=True)
@click.option("--begin", default='1970-01-01', help="Starting date (YYYY-MM-DD)", required=True)
@click.option("--end", default=None, help="End date (YYYY-MM-DD)")
@click.option("--max_cloud_cover", default=100, help="Maximum percentage of clouds (0-100)")
@click.option("--max_results", default=50000, help="Maximum number of search results")
@click.option("--tier", default='T1', help="Landsat collection tier")
@click.option("--ee_username", default=None, help="EarthExplorer username", required=True)
@click.option("--ee_passwd", default=None, help="EarthExplorer password", required=True)
@click.option("--save_id", default=None, help="Filename for saving order id")

def ls_espa_order(aoi, product, begin, end, max_cloud_cover, max_results,tier, ee_username, ee_passwd, save_id):
    """Order Landsat scenes at ESPA"""

    from utils.downloader import Espa, Usgs
    import geopandas as gpd
    import pandas as pd
    import datetime
    import sys

    ee_url   = "https://earthexplorer.usgs.gov/inventory/json/v/1.4.1/"

    aoi = gpd.read_file(aoi)
    left = aoi.bounds.minx[0]
    right = aoi.bounds.maxx[0]
    bottom = aoi.bounds.miny[0]
    top = aoi.bounds.maxy[0]

    sat_df= pd.DataFrame( columns=["sat", "sensor", "collection"])
    if 'landsat8' in product:
        sat_df = sat_df.append({'sat' : '8' , 'sensor' :"LC08", 'collection' : 'LANDSAT_8_C1'} , ignore_index=True)
    if 'landsat7' in product:
        sat_df = sat_df.append({'sat' : 'ETM' , 'sensor' :"LE07", 'collection' : 'LANDSAT_ETM_C1'} , ignore_index=True)
    if 'landsat5' in product:
        sat_df = sat_df.append({'sat' : 'TM' , 'sensor' :"LT05", 'collection' : 'LANDSAT_TM_C1'} , ignore_index=True)

    if sat_df.shape[0] == 0:
        sys.exit(product + ' is not a valid product')

    wrs = gpd.read_file('data/landsat/wrs2.shp')

    scenes_sel = gpd.overlay(wrs, aoi, how='intersection')
    scenes_sel['PATH'] = scenes_sel['PATH'].apply(lambda x: str(x).zfill(3))
    scenes_sel['ROW'] = scenes_sel['ROW'].apply(lambda x: str(x).zfill(3))
    pathrows = scenes_sel['PATH'] + scenes_sel['ROW']

    cl = Usgs(username=ee_username, password=ee_passwd)
    login = cl.login()

    if login == True:
        print('USGS: Succesfull login')
    else:
        sys.exit('USGS: Login failed')

    if end is None:
        end = datetime.date.today()
    begin = datetime.datetime.strptime(begin, "%Y-%M-%d")

    scenes_df = pd.DataFrame([])
    for s in sat_df['collection']:
         df = cl.search(s, bbox = (left,bottom, right ,top), begin=begin,end=end,
                        max_cloud_cover=max_cloud_cover, starting_number=1, max_results=max_results)
         scenes = df.loc[df['sensor'].isin(sat_df['sensor']) & df['pathrow'].isin(pathrows)]
         scenes_df=scenes_df.append( scenes )

    scenes_df['unikid'] = scenes_df.displayId.str[0:4] + scenes_df.displayId.str[10:16] + \
                            scenes_df.displayId.str[17:25] + scenes_df.displayId.str[35:37] + \
                            scenes_df.displayId.str[38:40]

    cl.logout()

    scenes_df = scenes_df.loc[scenes_df['tier'] == tier]

    ls=scenes_df['displayId'].to_list()

    # place order at ESPA and download when finished
    espa_order=Espa(username = ee_username, password=ee_passwd, scenelist = ls, out=None)
    order = espa_order.place_order()

    if save_id:
        text_file = open(save_id, "w")
        n = text_file.write(espa_order.orderid)
        text_file.close()


    click.echo("eo2cube ls_espa_order")


@eo2cube.command()
@click.option("--espa_order_id", default=None, help="ESPA order id", required=True)
@click.option("--download_dir", default=None, help="Define download directory", required=True)
@click.option("--ee_username", default=None, help="EarthExplorer username", required=True)
@click.option("--ee_passwd", default=None, help="EarthExplorer password", required=True)

def espa_download(espa_order_id, download_dir, ee_username, ee_passwd):
    """Download ESPA order"""

    from pathlib import Path
    from utils.downloader import Espa

    id_file = Path(espa_order_id)
    if id_file.is_file():
        f = open(id_file, "r")
        id = f.read()
    else:
        id = espa_order_id

    espa_order=Espa(username = ee_username, password=ee_passwd, scenelist=None, out=download_dir)
    espa_order.set_orderid(id)
    espa_complete  = espa_order.download(check_delay=20)

    click.echo("espa_download")

@eo2cube.command()
@click.option("--aoi", default=None, help="Shapefile defining the area of interest", required=True)
@click.option("--product", default=['landsat8'], help="Define which data to search for (either landsat5,landsat7 or landsat8)", required=True)
@click.option("--begin", default='1970-01-01', help="Starting date (YYYY-MM-DD)", required=True)
@click.option("--end", default=None, help="End date (YYYY-MM-DD)")
@click.option("--max_cloud_cover", default=100, help="Maximum percentage of clouds (0-100)")
@click.option("--max_results", default=50000, help="Maximum number of search results")
@click.option("--tier", default='T1', help="Landsat collection tier")
@click.option("--ee_username", default=None, help="EarthExplorer username", required=True)
@click.option("--ee_passwd", default=None, help="EarthExplorer password", required=True)
@click.option("--db_name", default=None, help="Database name", required=True)
@click.option("--db_user", default=None, help="Database username", required=True)
@click.option("--db_pass", default=None, help="Database password", required=True)
@click.option("--host", default="127.0.01", help="Host IP")
@click.option("--port", default="5432", help="Port")
@click.option("--save_id", default=None, help="Filename for saving order id")

def ls_dc_update(aoi, product, begin, end, max_cloud_cover, max_results,tier, ee_username, ee_passwd, db_name, db_user, db_pass, host, port, save_id):

    from utils.downloader import Espa, Usgs
    import geopandas as gpd
    import pandas as pd
    import datetime
    import psycopg2
    import sys

    ee_url   = "https://earthexplorer.usgs.gov/inventory/json/v/1.4.1/"

    aoi = gpd.read_file(aoi)
    left = aoi.bounds.minx[0]
    right = aoi.bounds.maxx[0]
    bottom = aoi.bounds.miny[0]
    top = aoi.bounds.maxy[0]

    sat_df= pd.DataFrame( columns=["sat", "sensor", "collection"])
    if 'landsat8' in product:
        sat_df = sat_df.append({'sat' : '8' , 'sensor' :"LC08", 'collection' : 'LANDSAT_8_C1'} , ignore_index=True)
    if 'landsat7' in product:
        sat_df = sat_df.append({'sat' : 'ETM' , 'sensor' :"LE07", 'collection' : 'LANDSAT_ETM_C1'} , ignore_index=True)
    if 'landsat5' in product:
        sat_df = sat_df.append({'sat' : 'TM' , 'sensor' :"LT05", 'collection' : 'LANDSAT_TM_C1'} , ignore_index=True)

    if sat_df.shape[0] == 0:
        sys.exit(product + ' is not a valid product')

    wrs = gpd.read_file('data/landsat/wrs2.shp')

    scenes_sel = gpd.overlay(wrs, aoi, how='intersection')
    scenes_sel['PATH'] = scenes_sel['PATH'].apply(lambda x: str(x).zfill(3))
    scenes_sel['ROW'] = scenes_sel['ROW'].apply(lambda x: str(x).zfill(3))
    pathrows = scenes_sel['PATH'] + scenes_sel['ROW']

    print('Searching for available Landsat scenes ...')
    cl = Usgs(username=ee_username, password=ee_passwd)
    login = cl.login()

    if login == True:
        print('USGS: Succesfull login')
    else:
        sys.exit('USGS: Login failed')

    if end is None:
        end = datetime.date.today()
    begin = datetime.datetime.strptime(begin, "%Y-%M-%d")

    scenes_df = pd.DataFrame([])
    for s in sat_df['collection']:
         df = cl.search(s, bbox = (left,bottom, right ,top), begin=begin,end=end,
                        max_cloud_cover=max_cloud_cover, starting_number=1, max_results=max_results)
         scenes = df.loc[df['sensor'].isin(sat_df['sensor']) & df['pathrow'].isin(pathrows)]
         scenes_df=scenes_df.append( scenes )

    scenes_df['unikid'] = scenes_df.displayId.str[0:4] + scenes_df.displayId.str[10:16] + \
                            scenes_df.displayId.str[17:25] + scenes_df.displayId.str[35:37] + \
                            scenes_df.displayId.str[38:40]

    cl.logout()

    scenes_df = scenes_df.loc[scenes_df['tier'] == tier]

    conn = psycopg2.connect(user=db_user, password=db_pass, host=host, port=port, database=db_name)

    sql_command = "SELECT DISTINCT (dataset_location.uri_body) AS platform FROM agdc.dataset_location WHERE ((archived IS NULL));"
    ingested = pd.read_sql(sql_command, conn)
    if ingested.empty != True:
        ingested_df = ingested.loc[
        ingested["platform"].str.contains(
            ".*?(landsat_c1).*?(datacube-metadata\\.yaml)"
        )
    ]
    ingested_n = (
        ingested_df["platform"]
        .str.split("/", expand=True)
        .rename(columns=lambda x: f"v{x+1}")
    )
    ingested_scenes = (
        ingested_n["v7"].str.split("-", expand=True).rename(columns=lambda x: f"v{x+1}")
    )

    scenes_df = scenes_df.loc[~scenes_df["unikid"].isin(ingested_scenes['v1'])]

    ls=scenes_df['displayId'].to_list()

    # place order at ESPA and download when finished
    espa_order=Espa(username = ee_username, password=ee_passwd, scenelist = ls, out=None)
    order = espa_order.place_order()

    if save_id:
        text_file = open(save_id, "w")
        n = text_file.write(espa_order.orderid)
        text_file.close()

    click.echo("eo2cube ls_dc_update")

@eo2cube.command()
@click.option("--download_dir", default=None, help="Define download directory to filter prior downloaded scenes", required=True)
@click.option("--aoi", default=None, help="Path to shapefile defining your area of interest", required=True)
@click.option("--list_only", default=False, help="Only returns a list of available scenes without downloading the data")
@click.option("--username", default=None,prompt=True, help="Copernicus Open Access Hub username", required=True)
@click.option('--password', prompt=True, hide_input=True, confirmation_prompt=True)
@click.option("--platform",default="S1A*",help="Define which data to search for (either S1A* for Sentinel-1A or S2A* for Sentinel-2A)")
@click.option("--start_date", default=None, help=" Define starting date of search (yyyy-mm-dd)", required=True)
@click.option("--end_date", default=None, help="Define ending date of search (yyyy-mm-dd)", required=True)
@click.option("--date_type",default="beginPosition",help="Define the type of the given dates (please select from beginPosition, endPosition and ingestionDate)")
@click.option("--min_overlap",default=0.1,help="Define minimum overlap (0-1) between area of interest and scene footprint (Default: 0)")
@click.option("--producttype",default="GRD",help="Define which product to search")

def s1_download(download_dir, aoi, list_only,username, password, platform, producttype, start_date, end_date, date_type, min_overlap):
    
    """Download Sentinel-1 scenes"""
    
    import geopandas as gpd
    import os
    from utils.downloader import  SentinelDownloader
    
    api_url = "https://scihub.copernicus.eu/apihub/"

    region = str(gpd.read_file(aoi).geometry[0])
    print(password)
    if not os.path.exists(download_dir):
        os.makedirs(download_dir)
    s1 = SentinelDownloader(username, password, api_url=api_url)
    s1.set_geometries(region)
    s1.set_download_dir(download_dir)
    s1.search(
        platform=platform,
        min_overlap=min_overlap,
        producttype=producttype,
        start_date=start_date,
        end_date=end_date,
        date_type=date_type
    )

    if list_only:
        s1.write_results(file_type="json", filename="sentinel_scenes.json")
    else:
        s1.write_results(file_type="wget", filename="sentinel_api_download")
        s1.download_all()

    click.echo("eo2cube s1_download")

@eo2cube.command()
@click.option("--input_dir", default=None, help="Directory containing the downloaded Sentinel1 GRD files", required=True)
@click.option("--db_name", default=None, help="Name of the database to store the Sentinel1 scenes", required=True)

def s1_create_archive(input_dir, db_name):
    """Create Sentinel-1 archive"""
    from pyroSAR import Archive, identify
    import glob

    dbfile = db_name

    files = glob.glob(input_dir + "/S1A*.zip")
    n = len(files)

    print('Found ' + str(n) + ' scenes')

    with Archive(dbfile) as archive:
        for name in files:
            scene = identify(name)
            archive.insert(scene)

    click.echo("eo2cube s1_create_archive")

@eo2cube.command()
@click.option("--archive", default=None, help="Path to the scene database", required=True)
#@click.option("--roi", default=None, help="A geometry with which the scenes need to overlap")
@click.option("--maxdate", default=None, help="The maximum acquisition date in format")
@click.option("--processdir", default=None, help="A directory to be scanned for already processed scenes; the selected scenes will be filtered to those that have not yet been processed")
@click.option("--recursive", default=False, help="Should also the subdirectories of the processdir be scanned?")
@click.option("--polarization", default=['VH', 'VV'], help="List of polarization strings, e.g. [‘HH’, ‘VV’]")
@click.option("--resolution", default=20, help="The target pixel resolution in meters")
@click.option("--sensor", default=('S1A', 'S1B'), help="Define which sensor to search for e.g ('S1A', 'S1B')")
@click.option("--product", default='GRD', help="Define which product to search for")
@click.option("--acquisition_mode", default='IW', help="Define aquisition mode to search for")
@click.option("--verbose", default=False, help="Print details about the selection including the SQL query?")

def s1_preprocessing(archive,  maxdate, processdir, sensor, product, resolution, recursive, polarization, acquisition_mode, verbose ):
    """Preprocess Sentinel-1 scenes"""
    from pyroSAR import Archive, identify
    from pyroSAR.snap import geocode
    from datetime import datetime

    archive = Archive(archive)

    if maxdate == None:
        maxdate = datetime.now().strftime("%Y%m%dT%H%M%S")

    selection_proc = archive.select(processdir=processdir,
                                    recursive=recursive,
                                    polarizations=polarization,
                                    maxdate=maxdate,
                                    sensor=sensor,
                                    product=product,
                                    acquisition_mode=acquisition_mode,
                                    verbose=verbose
                                    )
    archive.close()
    print(selection_proc)

    for scene in selection_proc:
        geocode(infile=scene, outdir=processdir, tr=resolution, scaling='db')


    click.echo("eo2cube s1_create_archive")

@eo2cube.command()
@click.option("--input_dir", default=None, help="Directory containing preprocessed scenes")
@click.option("--sensor", default=('S1A', 'S1B'), help="Define which sensor to search for e.g ('S1A', 'S1B')")
@click.option("--yml_index_outdir", default=None, help="Directory stroing the index yaml files")
@click.option("--yml_product", default=None, help="Define name of the product yaml")
@click.option("--acquisition_mode", default='IW', help="Define aquisition mode to search for")
@click.option("--product_name_indexed", default='S1_GRD_index', help="Define product name")
@click.option("--product_type", default='gamma0', help="Define product_type")
@click.option("--description", default='Gamma Naught RTC backscatter', help="Product description")

def s1_prepare(input_dir, sensor,yml_index_outdir, yml_product, acquisition_mode, product_name_indexed, product_type,description):
    """Prepare Sentinel-1 scenes for data cube ingestion"""
    from pyroSAR.datacube_util import Product, Dataset
    from pyroSAR.ancillary import find_datasets, groupby


    archive_s1 = input_dir
    scenes_s1 = find_datasets(archive_s1, sensor=sensor, acquisition_mode=acquisition_mode)
    grouped = groupby(scenes_s1, 'outname_base')
    units = {'VV': 'backscatter VV', 'VH': 'backscatter VH'}

    with Product(name=product_name_indexed,product_type=product_type, description=description) as prod:
        for dataset in grouped:
            with Dataset(dataset, units=units) as ds:
                print(ds)
                prod.add(ds)
                prod.export_indexing_yml(ds, yml_index_outdir)
    prod.write(yml_product)

    print(prod)

    click.echo("eo2cube s1_prepare")

@eo2cube.command()
@click.option("--yml_product", default=None, help="Path to the product yaml")
@click.option("--yml_ingest", default=None, help="Define name of the ingestion yaml")
@click.option("--product_name_ingested", default='S1_GRD_ingest', help="Define product ingestion name")
@click.option("--ingest_location", default=None, help="Define location for storing ingested files")

def s1_ingestion_yaml(yml_product,yml_ingest,product_name_ingested,ingest_location):
    """Create Sentinel-1 ingestion yaml"""

    from pyroSAR.datacube_util import Product

    with Product(yml_product) as prod:
        prod.export_ingestion_yml(yml_ingest, product_name_ingested, ingest_location,chunking={'x': 512, 'y': 512, 'time': 1})



    click.echo("eo2cube s1_ingestion_yaml")

@eo2cube.command()
@click.option("--output", default=None, help="Output path for the yaml", required=True)
@click.option("--processing_type", default='Sen2Cor', help="Type of preprocessing")

def s2_type_yaml(output, processing_type):
    """Create data type yaml files for Sentinel-2"""

    import yaml
    from utils.yaml_template import type_yaml, Ydumper


    if processing_type == 'Sen2Cor':
        data_type_yaml = type_yaml()
        if output[-5:] == '.yaml':
            with open(output, 'w') as yaml_file:
                yaml.dump(data_type_yaml, yaml_file, Dumper=Ydumper,  default_flow_style=False,  sort_keys=False)
        else:
            with open(output + '/s2_ard_scene.yaml', 'w') as yaml_file:
                yaml.dump(data_type_yaml,  yaml_file, Dumper=Ydumper, default_flow_style=False,  sort_keys=False)
    else:
        print('processing_type must be Sen2Cor')

    click.echo("eo2cube s2_type_yaml")

@eo2cube.command()
@click.option("--input_dir", default=None, help="Directory containing the prepared Level-2A Sentinel files", required=True)
@click.option("--output", default=None, help="Path to the output file", required=True)

def s2_ingestion_yaml(input_dir, output):

    """Create ingestion yaml for Sentinel-2"""

    from utils.yaml_template import ingestion_yaml, Ydumper
    import rasterio
    import glob

    output_type = click.prompt('Please insert the name of the data_type', default='s2_ard_scene')
    output_type = click.prompt('Please insert the name of the output_type', default='s2_l2a_project')
    description = click.prompt('Please insert a small project description', default='Sentinel 2 ARD (L2A) scenes')
    title = click.prompt('Please insert a project title', default='Open Data Cube - Sentinel 2 ARD')
    summary = click.prompt('Please insert a project title', default='Sentinel 2 MSI ARD ')
    institution = click.prompt('Please insert a institution name', default='')
    keywords = click.prompt('Please insert some kewords', default='')
    project = click.prompt('Please insert a project name', default='my_projects')
    tile_size = click.prompt('Please insert tile size', default=1)

    ingest_yaml = ingestion_yaml()
    ingest_yaml['output_type'] = output_type
    ingest_yaml['description'] = description
    ingest_yaml['global_attributes']['title'] = title
    ingest_yaml['global_attributes']['summary'] = summary
    ingest_yaml['global_attributes']['institution'] = institution
    ingest_yaml['global_attributes']['keywords'] = keywords
    ingest_yaml['global_attributes']['project'] = project

    files = glob.glob(input_dir + "/S2A_MSIL2A*/*_10m.tif")

    with rasterio.open(files[0]) as r:
        bbx = r.bounds

    extent=[bbx[0],bbx[1],bbx[2],bbx[3]]

    for file in files:
        with rasterio.open(file) as r:
            bbx = r.bounds

        if bbx[0] < extent[0]:
            extent[0]=bbx[0]
        if bbx[1] < extent[1]:
            extent[1]=bbx[1]
        if bbx[2] > extent[2]:
            extent[2]=bbx[2]
        if bbx[3] > extent[3]:
            extent[3]=bbx[3]

    ingest_yaml['ingestion_bounds']['left'] = extent[0]
    ingest_yaml['ingestion_bounds']['bottom'] = extent[1]
    ingest_yaml['ingestion_bounds']['right'] = extent[2]
    ingest_yaml['ingestion_bounds']['top'] = extent[3]

    resolution_x = []
    resolution_y = []
    for file in files:
        with rasterio.open(file) as r:
            resolution_x.append(r.res[0])
            resolution_y.append(r.res[1])

    mean_res_x = sum(resolution_x)/len(resolution_x)
    mean_res_y = sum(resolution_y)/len(resolution_y)
    m_res = (mean_res_x + mean_res_y)/2
    ts = (math.floor(tile_size / m_res))*(m_res)

    ingest_yaml['storage']['longitude'] = ts
    ingest_yaml['storage']['latitude'] = ts
    ingest_yaml['resolution']['longitude'] = m_res
    ingest_yaml['resolution']['latitude'] = m_res

    if output[-5:] == '.yaml':
        with open(output, 'w') as yaml_file:
            yaml.dump(ingest_yaml, yaml_file, Dumper=Ydumper, default_flow_style=False,  sort_keys=False)
    else:
        with open(output + output_type + '.yaml', 'w') as yaml_file:
            yaml.dump(ingest_yaml, yaml_file, Dumper=Ydumper, default_flow_style=False,  sort_keys=False)

    click.echo("eo2cube s2_ingestion_yaml")

@eo2cube.command()
@click.option("--download_dir", default=None, help="Define download directory to filter prior downloaded scenes", required=True)
@click.option("--search_only", default=False, help="Only returns a list of available scenes without downloading the them")
@click.option("--aoi", default=None, help="Path to shapefile defining your area of interest", required=True)
@click.option("--username", default=None,prompt=True, help="Copernicus Open Access Hub username", required=True)
@click.option('--password', prompt=True, hide_input=True, confirmation_prompt=True, required=True)
@click.option("--platform",default="S2A*",help="Define which data to search for (either S1A* for Sentinel-1A or S2A* for Sentinel-2A)")
@click.option("--start_date", default=None, help=" Define starting date of search (yyyy-mm-dd)", required=True)
@click.option("--end_date", default=None, help="Define ending date of search (yyyy-mm-dd)", required=True)
@click.option("--date_type",default="beginPosition",help="Define the type of the given dates (please select from beginPosition, endPosition and ingestionDate)")
@click.option("--min_overlap",default=0.1,help="Define minimum overlap (0-1) between area of interest and scene footprint (Default: 0)")
@click.option("--producttype",default="S2MSI1C",help="Define which product to search")

def s2_download(download_dir, aoi,search_only, username, password, platform, producttype, start_date, end_date, date_type, min_overlap):
    """Downlad Sentinel-2 scenes"""

    from utils.downloader import  SentinelDownloader
    import geopandas as gpd
    import os

    api_url = "https://scihub.copernicus.eu/apihub/"

    region = str(gpd.read_file(aoi).geometry[0])
    if not os.path.exists(download_dir):
        os.makedirs(download_dir)
    s2 = SentinelDownloader(username, password, api_url=api_url)
    s2.set_geometries(region)
    s2.set_download_dir(download_dir)
    s2.search(
        platform=platform,
        min_overlap=min_overlap,
        producttype=producttype,
        start_date=start_date,
        end_date=end_date,
        date_type=date_type
    )

    s2.write_results(file_type="wget", filename="sentinel_api_download")
    s2.download_all()

    click.echo("eo2cube s2_download")

@eo2cube.command()
@click.option("--folder", default=None, help="Path to file(s)", required=True)
@click.option("--output_dir", default=None, help="Output directory", required=True)
@click.option("--cores", default=2, help="Number of cores to use")

def unzip(folder ,output_dir, cores):
    """Unzip files"""

    import multiprocessing
    from functools import partial
    from utils.helper import unzipper, createFolder
    import glob
    from tqdm import tqdm

    exfiles = glob.glob(folder + '/*')
    n = len(exfiles)
    if n < 1:
        sys.exit('No .zip or tar.gz files found ')

    createFolder(output_dir)

    print('Unzipping files ...')
    print('Found ' + str(n) + ' scenes')

    with multiprocessing.Pool(cores) as pool:
        MAX_COUNT = len(exfiles)
        func = partial(unzipper, output_dir = output_dir)
        for res in tqdm(pool.imap(func, exfiles),total=MAX_COUNT):
            if res is not None:
                print(res['sum'])
        pool.close()
        pool.join()

    click.echo("eo2cube unzip")


@eo2cube.command()
@click.option("--l2a_process", default=None, help="Path to L2A_Process", required=True)
@click.option("--input_dir", default=None, help="Directory containing Level-1C data", required=True)
@click.option("--cores", default=2, help="Number of cores to use")

def sen2cor(input_dir, l2a_process, cores):
    """Wrapper for Sen2Cor"""

    import multiprocessing
    from utils.helper import sen2cor_single
    import glob
    from tqdm import tqdm
    from functools import partial

    files = glob.glob(input_dir + "/S2A_MSIL1C*")
    n = len(files)
    print('Found ' + str(n) + ' scenes')
    with multiprocessing.Pool(cores) as pool:
        MAX_COUNT = len(files)
        func = partial(sen2cor_single, l2a_path = l2a_process)
        for res in tqdm(pool.imap(func, files),total=MAX_COUNT):
            if res is not None:
                print(res['sum'])
        pool.close()
        pool.join()

    click.echo("eo2cube sen2cor")

@eo2cube.command()

@click.option("--input_dir", default=None, help="Directory containing Level-2A data", required=True)
@click.option("--config_yaml", default=None, help="Path to the data_type yaml", required=True)
@click.option("--clean", default=False, help="If False keep SAFE files else remove them")
@click.option("--cores", default=2, help="Number of cores to use")

def s2_jpg2tiff(input_dir, config_yaml, clean, cores):
    """Convert JPEG2000 files to GeoTiff"""

    import multiprocessing
    from functools import partial
    from utils.helper import jp2_to_tif
    import glob
    from tqdm import tqdm
    import yaml
    import nested_lookup

    print('Converting to tiff ... ')
    stream = open(config_yaml, 'r')
    yaml_dict = yaml.load(stream,Loader=yaml.FullLoader)
    blist = nested_lookup.nested_lookup('name',yaml_dict)
    print('Bands: ')
    print(blist)
    files = glob.glob(input_dir + "/S2A_MSIL2A*.SAFE")
    n = len(files)
    print('Found ' + str(n) + ' scenes')

    with multiprocessing.Pool(cores) as pool:
        MAX_COUNT = len(files)
        func = partial(jp2_to_tif,  path =input_dir, clean=clean, bands = blist)
        for res in tqdm(pool.imap(func, files),total=MAX_COUNT):
            if res is not None:
                print(res['sum'])
        pool.close()
        pool.join()

    click.echo("s2_jpg2tiff")

@eo2cube.command()
@click.option("--input_dir", default=None, help="Directory containing the data to reproject", required=True)
@click.option("--output_dir", default=None, help="Output directory", required=True)
@click.option("--cores", default=2, help="Number of cores to use")

def reproject(input_dir,output_dir,cores):

    """Reproject scenes"""

    import multiprocessing
    from functools import partial
    from utils.helper import to_wgs84, createFolder
    import glob
    from tqdm import tqdm

    print('Reprojecting to WGS84 ...')
    files = glob.glob(input_dir + "/*.tif")
    n=len(files)
    createFolder(output_dir)
    print('Found ' + str(n) + ' scenes')
    with multiprocessing.Pool(cores) as pool:
        MAX_COUNT = len(files)
        func = partial(to_wgs84,  out = output_dir)
        for res in tqdm(pool.imap(func, files),total=MAX_COUNT):
            if res is not None:
                print(res['sum'])

    click.echo("reproject")

@eo2cube.command()
@click.option("--db_name", default=None, help="Database name")
@click.option("--db_user", default=None, help="Database username")
@click.option("--db_pass", default=None, help="Database password")
@click.option("--product", default=None, help="Product name")
@click.option("--host", default="127.0.01", help="Host IP")
@click.option("--port", default="5432", help="Port")
@click.option("--schema", default="agdc", help="Database schema")

def remove_product(db_name, db_user,db_pass, product, host, port, schema):

    """Remove product from database"""

    import psycopg2

    product=str(product)
    conn = psycopg2.connect(user=db_user, password=db_pass, host=host, port=port, database=db_name, options=f'-c search_path={schema}')
    cur = conn.cursor()
    sql_command = "SELECT count(*), t.name FROM dataset  LEFT JOIN dataset_type t ON dataset.dataset_type_ref = t.id GROUP BY t.name;"
    cur.execute("WITH datasets as (SELECT id FROM dataset where dataset.dataset_type_ref = (select id FROM dataset_type WHERE name =  %s)) DELETE FROM dataset_source USING datasets where dataset_source.dataset_ref = datasets.id;", (product,))

    cur.execute("WITH datasets as (SELECT id FROM dataset where dataset.dataset_type_ref = (select id FROM dataset_type WHERE name =  %s)) DELETE FROM dataset_source USING datasets where dataset_source.dataset_ref = datasets.id;", (product,))
    cur.execute("WITH datasets as (SELECT id FROM dataset where dataset.dataset_type_ref = (select id FROM dataset_type WHERE name = %s)) DELETE FROM dataset_location USING datasets where dataset_location.dataset_ref = datasets.id;", (product,))
    cur.execute("DELETE FROM dataset where dataset.dataset_type_ref = (select id from dataset_type where dataset_type.name =  %s);", (product,))

    cur.execute("DELETE FROM dataset_type where dataset_type.name = %s;", (product,))
    conn.commit()
    cur.close()
    conn.close()

    click.echo("eo2cube remove_product")

@eo2cube.command()
@click.option("--db_name", default=None, help="Database name")
@click.option("--db_user", default=None, help="Database username")
@click.option("--db_pass", default=None, help="Database password")
@click.option("--product", default=None, help="Product name")
@click.option("--host", default="127.0.01", help="Host IP")
@click.option("--port", default="5432", help="Port")
@click.option("--schema", default="agdc", help="Database schema")


def remove_dataset(db_name, db_user,db_pass, product, host, port, schema):

    """Remove ingested dataset from database"""

    product=str(product)
    conn = psycopg2.connect(user=db_user, password=db_pass, host=host, port=port, database=db_name, options=f'-c search_path={schema}')
    cur = conn.cursor()
    sql_command = "SELECT count(*), t.name FROM dataset  LEFT JOIN dataset_type t ON dataset.dataset_type_ref = t.id GROUP BY t.name;"
    cur.execute("WITH datasets as (SELECT id FROM dataset where dataset.dataset_type_ref = (select id FROM dataset_type WHERE name =  %s)) DELETE FROM dataset_source USING datasets where dataset_source.dataset_ref = datasets.id;", (product,))

    cur.execute("WITH datasets as (SELECT id FROM dataset where dataset.dataset_type_ref = (select id FROM dataset_type WHERE name =  %s)) DELETE FROM dataset_source USING datasets where dataset_source.dataset_ref = datasets.id;", (product,))
    cur.execute("WITH datasets as (SELECT id FROM dataset where dataset.dataset_type_ref = (select id FROM dataset_type WHERE name = %s)) DELETE FROM dataset_location USING datasets where dataset_location.dataset_ref = datasets.id;", (product,))
    cur.execute("DELETE FROM dataset where dataset.dataset_type_ref = (select id from dataset_type where dataset_type.name =  %s);", (product,))

    cur.execute("DELETE FROM dataset_type where dataset_type.name = %s;", (product,))
    conn.commit()
    cur.close()
    conn.close()
    click.echo("eo2cube remove_dataset")


if __name__ == "__main__":
    eo2cube()
