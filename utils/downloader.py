import os
import sys
from datetime import datetime, date
from shapely.wkt import loads
import requests
import progressbar as pb
import zipfile as zf
import json
from pandas.io.json import json_normalize
import pandas as pd
import time
import multiprocessing
import concurrent.futures
import re
from tqdm import tqdm


class EarthExplorer(object):

    def __init__(self, username, password, apiKey=None, maxthreads=5):
        self.maxthreads = maxthreads
        self.USER = username
        self.PASSWORD = password
        self.endpoint = 'https://m2m.cr.usgs.gov/api/api/json/stable/'
        self.api_key = apiKey

    def sendRequest(self, url, data, apiKey=None, exitIfNoResponse=True):
        json_data = json.dumps(data)
        apiKey = self.api_key
        if apiKey is None:
            response = requests.post(url, json_data)
        else:
            headers = {'X-Auth-Token': apiKey}
            response = requests.post(url, json_data, headers=headers)
        try:
            httpStatusCode = response.status_code
            if response is None:
                print("No output from service")
                if exitIfNoResponse:
                    sys.exit()
                else:
                    return False
            output = json.loads(response.text)

            if output['errorCode'] is not None:
                print(output['errorCode'], "- ", output['errorMessage'])
                if exitIfNoResponse:
                    sys.exit()
                else:
                    return False

            if httpStatusCode == 404:
                print("404 Not Found")
                if exitIfNoResponse:
                    sys.exit()
                else:
                    return False

            elif httpStatusCode == 401:
                print("401 Unauthorized")
                if exitIfNoResponse:
                    sys.exit()
                else:
                    return False

            elif httpStatusCode == 400:
                print("Error Code", httpStatusCode)
                if exitIfNoResponse:
                    sys.exit()
                else:
                    return False

        except Exception as e:
            response.close()
            print(e)
            if exitIfNoResponse:
                sys.exit()
            else:
                return False
        response.close()
        return output['data']

    def login(self):
        url = self.endpoint + "login"
        payload = {'username': self.USER, 'password': self.PASSWORD}
        self.api_key = self.sendRequest(url=url, data=payload)

    def logout(self):
        logout_endpoint = self.endpoint + 'logout'
        if sendRequest(logout_endpoint, None, self.api_key) is None:
            print("Logged Out\n")
        else:
            print("Logout Failed\n")

    def search(self, collection, bbox, begin=None, end=None,
               max_cloud_cover=100, starting_number=1,tier=None, max_results=50000):

        begin = datetime.strptime(begin, "%Y-%m-%d")
        end = datetime.strptime(end, "%Y-%m-%d")
        self.collection = collection
        search_endpoint = self.endpoint + 'scene-search'
        params = {"datasetName": collection,
                  "sceneFilter": {
                    "acquisitionFilter": {
                        "end": end.isoformat(),
                        "start": begin.isoformat()
                    },
                    "spatialFilter": {
                         "filterType": "mbr",
                         "lowerLeft": {
                                "latitude": bbox[1],
                                "longitude": bbox[0]},
                         "upperRight": {
                                "latitude": bbox[3],
                                "longitude": bbox[2]},
                    },
                    "cloudCoverFilter": {
                        "max": max_cloud_cover,
                        "min": 0,
                        },
                    },
                  "maxResults": max_results,
                  "startingNumber": starting_number
                  }
        r = self.sendRequest(url=search_endpoint, data=params,
                             apiKey=self.api_key, exitIfNoResponse=True)
    
        self.results = pd.json_normalize(r['results'])
        if tier is not None and collection != 'sentinel_2a':
            if self.results.empty:
                print('No datasets found')
                sys.exit()
            else:
                self.results['tier'] = self.results['displayId'].str.split("_", expand=True)[6]
                self.results = self.results.loc[self.results['tier'] == tier]

        nrows = len(self.results.index)
        print(str(nrows) + " scenes found")

    def get_results(self):
        return self.results

    def thread_download(self, url_list):
        try:
            with concurrent.futures.ThreadPoolExecutor(self.maxthreads) as executor:
                fs = [executor.submit(self.downloadFile, url) for url in url_list]
                #print(fs.result())
        except KeyboardInterrupt:
            try:
                sys.exit(0)
            except SystemExit:
                os._exit(0)
        except Exception as e:
            pass
            print("Unknown error occured", e)
            raise
            sys.exit()

    def downloadFile(self, url):
        try:
            response = requests.get(url, stream=True)
            file_size = int(response.headers['Content-Length'])
            disposition = response.headers['content-disposition']
            filename = re.findall("filename=(.+)", disposition)[0].strip("\"")
            if self.path != "" and self.path[-1] != "/":
                filename = "/" + filename
            if os.path.exists(self.path+filename):
                first_byte = os.path.getsize(self.path+filename)
            else:
                first_byte = 0
            if first_byte >= file_size:
                return file_size
            print(f"Downloading {filename} ...\n")
            pbar = tqdm(
                total=file_size, initial=first_byte,
                unit='B', unit_scale=True, desc=filename)
            with(open(self.path+filename, 'wb')) as f:
                for chunk in response.iter_content(chunk_size=1024):
                    if chunk:
                        f.write(chunk)
                        pbar.update(1024)
            pbar.close()
        except Exception as e:
            print(f"Failed to download from {url}. Will try to re-download.")
            print(e)

    def download(self, path, maxthreads=2):

        self.path = path
        startTime = time.time()
        entityIds = self.results['entityId'].tolist()
        datasetName = self.collection
        idField = 'entityId'
        listId = f"temp_{datasetName}_list"
        payload = {"listId": listId, 'idField': idField,
                   "entityIds": entityIds, "datasetName": datasetName}

        count = self.sendRequest(self.endpoint + "scene-list-add", payload,
                                 self.api_key)
        print("Added", count, "scenes\n")

        payload = {"listId": listId, "datasetName": datasetName}

        products = self.sendRequest(self.endpoint + "download-options",
                                    payload, self.api_key)

        print("Got product download options\n")

        downloads = []
        for product in products:
            if product["bulkAvailable"]:
                downloads.append({"entityId": product["entityId"], "productId": product["id"]})
        payload = {"listId": listId}
        self.sendRequest(self.endpoint + "scene-list-remove",
                         payload, self.api_key)

        label = datetime.now().strftime("%Y%m%d_%H%M%S")
        payLoad = {"downloads": downloads, "label": label,
                   'returnAvailable': True}

        print(f"Sending download request ...\n")
        results = self.sendRequest(self.endpoint + "download-request",
                                   payLoad, self.api_key)
        print(f"Done sending download request\n")

        result_url = []
        for result in results['availableDownloads']:
            print(f"Get download url: {result['url']}\n")
            result_url.append(result['url'])

        self.threads = min(maxthreads, len(result_url))
        self.thread_download(url_list=result_url)

        preparingDownloadCount = len(results['preparingDownloads'])
        preparingDownloadIds = []
        if preparingDownloadCount > 0:
            for result in results['preparingDownloads']:
                preparingDownloadIds.append(result['downloadId'])

            payload = {"label": label}
            print("Retrieving download urls...\n")
            results = self.sendRequest(self.endpoint + "download-retrieve",
                                       payload, self.api_key, False)
            if results is not False:
                result_url = []
                for result in results['available']:
                    if result['downloadId'] in preparingDownloadIds:
                        preparingDownloadIds.remove(result['downloadId'])
                        print(f"Get download url: {result['url']}\n")
                        result_url.append(result['url'])

                self.threads = min(maxthreads, len(result_url))
                self.thread_download(url_list=result_url)

                result_url = []
                for result in results['requested']:
                    if result['downloadId'] in preparingDownloadIds:
                        preparingDownloadIds.remove(result['downloadId'])
                        print(f"Get download url: {result['url']}\n")
                        result_url.append(result['url'])

                self.threads = min(maxthreads, len(result_url))
                self.thread_download(url_list=result_url)

            while len(preparingDownloadIds) > 0:
                print(f"{len(preparingDownloadIds)} downloads are not available yet. Waiting for 30s to retrieve again\n")
                time.sleep(30)
                results = self.sendRequest(self.endpoint + "download-retrieve", payload, self.api_key, False)
                if results != False:
                    result_url = []
                    for result in results['available']:
                        if result['downloadId'] in preparingDownloadIds:
                            preparingDownloadIds.remove(result['downloadId'])
                            print(f"Get download url: {result['url']}\n" )
                            result_url.append(result['url'])

                    self.threads = min(maxthreads, len(result_url))
                    self.thread_download(url_list=result_url)

        print("Complete Downloading")
        executionTime = round((time.time() - startTime), 2)
        print(f'Total time: {executionTime} seconds')


class SentinelDownloader(object):
    """
    Sentinel Search & Download API
    Authors: Jonas Eberle <jonas.eberle@uni-jena.de>, Felix Cremer <felix.cremer@uni-jena.de>, John Truckenbrodt <john.truckenbrodt@uni-jena.de>
    Libraries needed: Shapely, GDAL/OGR, JSON, Progressbar, Zipfile, Datetime, Requests
    Example usage: Please see the "main" function at the end of this file
    """

    __esa_username = None
    __esa_password = None
    __esa_api_url = None

    __geometries = []
    __scenes = []
    __download_dir = "./"
    __data_dirs = []

    def __init__(
        self, username, password, api_url="https://scihub.copernicus.eu/apihub/"
    ):
        self.__esa_api_url = api_url
        self.__esa_username = username
        self.__esa_password = password

    def set_download_dir(self, download_dir):
        """Set directory for check against existing downloaded files and as directory where to download
        Args:
            download_dir: Path to directory
        """
        print("Setting download directory to %s" % download_dir)
        if not os.path.exists(download_dir):
            os.makedirs(download_dir)

        self.__download_dir = download_dir

    def set_data_dir(self, data_dir):
        """Set directory for check against existing downloaded files; this can be repeated multiple times to create a list of data directories
        Args:
            data_dir: Path to directory
        """
        print("Adding data directory {}".format(data_dir))
        self.__data_dirs.append(data_dir)

    def set_geometries(self, geometries):
        """Manually set one or more geometries for data search
        Args:
            geometries: String or List representation of one or more Wkt Geometries,
                Geometries have to be in Lat/Lng, EPSG:4326 projection!
        """
        # print('Set geometries:')
        # print(geometries)
        if isinstance(geometries, list):
            self.__geometries = geometries

        elif isinstance(geometries, str):
            self.__geometries = [geometries]

        else:
            raise Exception("geometries parameter needs to be a list or a string")

        # Test first geometry
        try:
            loads(self.__geometries[0])
        except Exception as e:
            print("The first geometry is not valid! Error: %s" % e)

    def get_geometries(self):
        """Return list of geometries"""
        return self.__geometries

    def load_sites(self, input_file, verbose=False):
        """Load features from input file and transform geometries to Lat/Lon (EPSG 4326)
        Args:
            input_file: Path to file that can be read by OGR library
            verbose: True if extracted geometries should be printed to console (default: False)
        """
        print("===========================================================")
        print("Loading sites from file %s" % input_file)

        if not os.path.exists(input_file):
            raise Exception("Input file does not exist: %s" % input_file)

        source = ogr.Open(input_file, 0)
        layer = source.GetLayer()

        in_ref = layer.GetSpatialRef()
        out_ref = osr.SpatialReference()
        out_ref.ImportFromEPSG(4326)

        coord_transform = osr.CoordinateTransformation(in_ref, out_ref)
        geometries = []

        for feature in layer:
            geom = feature.GetGeometryRef()
            geom.Transform(coord_transform)
            geom = geom.ExportToWkt()
            if verbose:
                print(geom)
            geometries.append(geom)

        self.__geometries = geometries
        print("Found %s features" % len(geometries))

    def search(
        self,
        platform,
        min_overlap=0,
        download_dir=None,
        start_date=None,
        end_date=None,
        date_type="beginPosition",
        **keywords
    ):
        """Search in ESA Data Hub for scenes with given arguments
        Args:
            platform: Define which data to search for (either 'S1A*' for Sentinel-1A or 'S2A*' for Sentinel-2A)
            min_overlap: Define minimum overlap (0-1) between area of interest and scene footprint (Default: 0)
            download_dir: Define download directory to filter prior downloaded scenes (Default: None)
            startDate: Define starting date of search (Default: None, all data)
            endDate: Define ending date of search (Default: None, all data)
            dataType: Define the type of the given dates (please select from 'beginPosition', 'endPosition', and
                'ingestionDate') (Default: beginPosition)
            **keywords: Further OpenSearch arguments can be passed to the query according to the ESA Data Hub Handbook
                (please see https://scihub.copernicus.eu/twiki/do/view/SciHubUserGuide/3FullTextSearch#Search_Keywords)
                missing under this link:
                - slicenumber: the graticule along an orbit; particularly important for interferometric applications
                    to identify overlapping scene pairs
        Mandatory args:
            platform
        Example usage:
            s1.search('S1A*', min_overlap=0.5, productType='GRD')
        """
        print("===========================================================")
        print("Searching data for platform %s" % platform)
        if platform not in ["S1A*", "S1B*", "S2A*", "S2B*", "S3A*", "S3B*"]:
            raise Exception(
                "platform parameter has to be S1A*, S1B*, S2A*, S2B*, S3A* or S3B*"
            )

        if download_dir is not None:
            self.set_download_dir(download_dir)

        date_filtering = ""
        if start_date is not None or end_date is not None:
            if start_date is None:
                raise Exception("Please specify also a starting date!")
            if end_date is None:
                end_date = datetime.now()
            if date_type not in ["beginPosition", "endPosition", "ingestionDate"]:
                raise Exception(
                    "dateType parameter must be one of beginPosition, endPosition, ingestionDate"
                )
            if isinstance(start_date, (datetime, date)):
                start_date = start_date.strftime("%Y-%m-%dT%H:%M:%S.%fZ")
            else:
                start_date = datetime.strptime(start_date, "%Y-%m-%d").strftime(
                    "%Y-%m-%dT%H:%M:%S.%fZ"
                )
            if isinstance(end_date, (datetime, date)):
                end_date = end_date.strftime("%Y-%m-%dT%H:%M:%S.%fZ")
            else:
                end_date = datetime.strptime(
                    end_date + " 23:59:59.999", "%Y-%m-%d %H:%M:%S.%f"
                ).strftime("%Y-%m-%dT%H:%M:%S.%fZ")
            date_filtering = " AND %s:[%s TO %s]" % (date_type, start_date, end_date)

        for geom in self.__geometries:
            print("===========================================================")

            index = 0
            scenes = []
            while True:
                url = self._format_url(
                    index, geom, platform, date_filtering, **keywords
                )
                print("Search URL: %s" % url)
                subscenes = self._search_request(url)
                if len(subscenes) > 0:
                    print(
                        "found %s scenes on page %s"
                        % (len(subscenes), index // 100 + 1)
                    )
                    scenes += subscenes
                    index += 100
                    print("=============================")
                if len(subscenes) < 100:
                    break

            print("%s scenes after initial search" % len(scenes))
            if len(scenes) > 0:
                scenes = self._filter_existing(scenes)
                scenes = self._filter_overlap(scenes, geom, min_overlap)
                print("%s scenes after filtering before merging" % len(scenes))
                self.__scenes = self._merge_scenes(self.__scenes, scenes)

        print("===========================================================")
        print("%s total scenes after merging" % len(self.__scenes))
        print("===========================================================")

    def get_scenes(self):
        """Return searched and filtered scenes"""
        return self.__scenes

    def print_scenes(self):
        """Print title of searched and filtered scenes"""

        def sorter(x):
            return re.findall("[0-9T]{15}", x)[0]

        titles = sorted([x["title"] for x in self.__scenes], key=sorter)
        print("\n".join(titles))

    def write_results(self, file_type, filename, output=False):
        """Write results to disk in different kind of formats
        Args:
            file_type: Use 'wget' to write download bash file with wget software, 'json' to write the dictionary object
                to file, or 'url' to write a file with downloadable URLs
            path: Path to file
            output: If True the written file will also be send to stdout (Default: False)
        """
        if file_type == "wget":
            self._write_download_wget(filename)
        elif file_type == "json":
            self._write_json(filename)
        else:
            self._write_download_urls(filename)

        if output:
            with open(filename, "r") as infile:
                print(infile.read())

    def download_all(self, download_dir=None):
        """Download all scenes
        Args:
            download_dir: Define a directory where to download the scenes
                (Default: Use default from class -> current directory)
        Returns:
            Dictionary of failed ('failed') and successfully ('success') downloaded scenes
        """
        if download_dir is None:
            download_dir = self.__download_dir

        downloaded = []
        downloaded_failed = []

        for scene in self.__scenes:
            url = scene["url"]
            filename = scene["title"] + ".zip"
            path = os.path.join(download_dir, filename)
            print("===========================================================")
            print("Download file path: %s" % path)

            try:
                response = requests.get(
                    url, auth=(self.__esa_username, self.__esa_password), stream=True
                )
            except requests.exceptions.ConnectionError:
                print("Connection Error")
                continue
            if "Content-Length" not in response.headers:
                print("Content-Length not found")
                print(url)
                continue
            size = int(response.headers["Content-Length"].strip())
            if size < 1000000:
                print("The found scene is to small: %s (%s)" % (scene["title"], size))
                print(url)
                continue

            print(
                "Size of the scene: %s MB" % (size / 1024 / 1024)
            )  # show in MegaBytes
            my_bytes = 0
            widgets = [
                "Downloading: ",
                pb.Bar(marker="*", left="[", right=" "),
                pb.Percentage(),
                " ",
                pb.FileTransferSpeed(),
                "] ",
                " of {0}MB".format(str(round(size / 1024 / 1024, 2))[:4]),
            ]
            pbar = pb.ProgressBar(widgets=widgets, maxval=size).start()

            try:
                down = open(path, "wb")
                for buf in response.iter_content(1024):
                    if buf:
                        down.write(buf)
                        my_bytes += len(buf)
                        pbar.update(my_bytes)
                pbar.finish()
                down.close()
            except KeyboardInterrupt:
                print(
                    "\nKeyboard interruption, remove current download and exit execution of script"
                )
                os.remove(path)
                sys.exit(0)

            # Check if file is valid
            print("Check if file is valid: ")
            valid = self._is_valid(path)

            if not valid:
                downloaded_failed.append(path)
                print("invalid file is being deleted.")
                os.remove(path)
            else:
                downloaded.append(path)

        return {"success": downloaded, "failed": downloaded_failed}

    def _is_valid(self, zipfile, minsize=1000000):
        """
        Test whether the downloaded zipfile is valid
        Args:
            zipfile: the file to be tested
            minsize: the minimum accepted file size
        Returns: True if the file is valid and False otherwise
        """
        if not os.path.getsize(zipfile) > minsize:
            print(
                "The downloaded scene is too small: {}".format(
                    os.path.basename(zipfile)
                )
            )
            return False
        try:
            archive = zf.ZipFile(zipfile, "r")
            try:
                corrupt = True if archive.testzip() else False
            except zlib.error:
                corrupt = True
            archive.close()
        except zf.BadZipfile:
            corrupt = True
        if corrupt:
            print(
                "The downloaded scene is corrupt: {}".format(os.path.basename(zipfile))
            )
        else:
            print("file seems to be valid.")
        return not corrupt

    def _format_url(
        self, startindex, wkt_geometry, platform, date_filtering, **keywords
    ):
        """Format the search URL based on the arguments
        Args:
            wkt_geometry: Geometry in Wkt representation
            platform: Satellite to search in
            dateFiltering: filter of dates
            **keywords: Further search parameters from ESA Data Hub
        Returns:
            url: String URL to search for this data
        """
        geom = loads(wkt_geometry)
        bbox = geom.envelope

        query_area = ' AND (footprint:"Intersects(%s)")' % bbox
        filters = ""
        for kw in sorted(keywords.keys()):
            filters += " AND (%s:%s)" % (kw, keywords[kw])

        url = os.path.join(
            self.__esa_api_url,
            "search?format=json&rows=100&start=%s&q=%s%s%s%s"
            % (startindex, platform, date_filtering, query_area, filters),
        )
        return url

    def _search_request(self, url):
        """Do the HTTP request to ESA Data Hub
        Args:
            url: HTTP URL to request
        Returns:
            List of scenes (result from _parseJSON method), empty list if an error occurred
        """
        try:
            content = requests.get(
                url, auth=(self.__esa_username, self.__esa_password), verify=True
            )
            if not content.status_code // 100 == 2:
                print(
                    "Error: API returned unexpected response {}:".format(
                        content.status_code
                    )
                )
                print(content.text)
                return []
            return self._parse_json(content.json())

        except requests.exceptions.RequestException as exc:
            print("Error: {}".format(exc))
            return []

    def _parse_json(self, obj):
        """Parse the JSON result from ESA Data Hub and create a dictionary for each scene
        Args:
            obj: Dictionary (if 1 scene) or list of scenes
        Returns:
            List of scenes, each represented as a dictionary
        """
        if "entry" not in obj["feed"]:
            print("No results for this feed")
            return []

        scenes = obj["feed"]["entry"]
        if not isinstance(scenes, list):
            scenes = [scenes]
        scenes_dict = []
        for scene in scenes:
            item = {
                "id": scene["id"],
                "title": scene["title"],
                "url": scene["link"][0]["href"],
            }

            for data in scene["str"]:
                item[data["name"]] = data["content"]

            for data in scene["date"]:
                item[data["name"]] = data["content"]

            for data in scene["int"]:
                item[data["name"]] = data["content"]

            scenes_dict.append(item)

        return scenes_dict

    def _filter_existing(self, scenes):
        """Filter scenes based on existing files in the define download directory and all further data directories
        Args:
            scenes: List of scenes to be filtered
        Returns:
            Filtered list of scenes
        """
        filtered = []
        dirs = self.__data_dirs + [self.__download_dir]
        for scene in scenes:
            exist = [
                os.path.isfile(os.path.join(dir, scene["title"] + ".zip"))
                for dir in dirs
            ]
            if not any(exist):
                filtered.append(scene)
        return filtered

    def _filter_overlap(self, scenes, wkt_geometry, min_overlap=0):
        """Filter scenes based on the minimum overlap to the area of interest
        Args:
            scenes: List of scenes to filter
            wkt_geometry: Wkt Geometry representation of the area of interest
            min_overlap: Minimum overlap (0-1) in decimal format between scene geometry and area of interest
        Returns:
            Filtered list of scenes
        """
        site = loads(wkt_geometry)

        filtered = []

        for scene in scenes:
            footprint = loads(scene["footprint"])
            intersect = site.intersection(footprint)
            overlap = intersect.area / site.area
            if overlap > min_overlap or (
                site.area / footprint.area > 1
                and intersect.area / footprint.area > min_overlap
            ):
                scene["_script_overlap"] = overlap * 100
                filtered.append(scene)

        return filtered

    def _merge_scenes(self, scenes1, scenes2):
        """Merge scenes from two different lists using the 'id' keyword
        Args:
            scenes1: List of prior available scenes
            scenes2: List of new scenes
        Returns:
            Merged list of scenes
        """
        existing_ids = []
        for scene in scenes1:
            existing_ids.append(scene["id"])

        for scene in scenes2:
            if not scene["id"] in existing_ids:
                scenes1.append(scene)

        return scenes1

    def _write_json(self, path):
        """Write JSON representation of scenes list to file
        Args:
            file: Path to file to write in
        """
        with open(path, "w") as outfile:
            json.dump(self.__scenes, outfile)
        return True

    def _write_download_wget(self, path):
        """Write bash file to download scene URLs based on wget software
        Please note: User authentication to ESA Data Hub (username, password) is being stored in plain text!
        Args:
            file: Path to file to write in
        """
        with open(path, "w") as outfile:
            for scene in self.__scenes:
                outfile.write(
                    'wget -c -T120 --no-check-certificate --user=%s --password=%s -O %s%s.zip "%s"\n'
                    % (
                        self.__esa_username,
                        self.__esa_password,
                        self.__download_dir,
                        scene["title"],
                        scene["url"].replace("$", "\$"),
                    )
                )
        return None

    def _write_download_urls(self, path):
        """Write URLs of scenes to text file
        Args:
            file: Path to file to write in
        """
        with open(path, "w") as outfile:
            for scene in self.__scenes:
                outfile.write(scene["url"] + "\n")
        return path

class Usgs(object):
    '''
     Adapted from Loïc Dutrieux
     Source : https://github.com/loicdtx/lsru
    '''
    def __init__(self, username, password, version='stable'):
        self.USER = username
        self.PASSWORD = password
        self.endpoint = '/'.join(['https://earthexplorer.usgs.gov/inventory/json/v',version])

    def login(self):
        login_endpoint = '/'.join([self.endpoint, 'login'])
        r = requests.post(login_endpoint,data={'jsonRequest': json.dumps({'username': self.USER,'password': self.PASSWORD})})
        if r.json()['errorCode'] is not None:
            return False
        self.api_key = r.json()['data']
        return True

    def search(self, collection, bbox, begin=None, end=None, max_cloud_cover=100,
               starting_number=1, max_results=50000):
        search_endpoint = '/'.join([self.endpoint, 'search'])
        params = {'apiKey': self.api_key,
                  'node': 'EE',
                  'datasetName': collection,
                  'maxCloudCover': max_cloud_cover,
                  'lowerLeft': {'latitude': bbox[1],
                                'longitude': bbox[0]},
                  'upperRight': {'latitude': bbox[3],
                                 'longitude': bbox[2]},
                  'maxResults': max_results,
                  'startingNumber': starting_number}
        if begin is not None:
            params.update(startDate=begin.isoformat())
        if end is not None:
            params.update(endDate=end.isoformat())
        r = requests.post(search_endpoint,
                          data={'jsonRequest': json.dumps(params)})
        rdat = r.json()['data']
        df = json_normalize(rdat['results'])
        df['sensor'] = df.displayId.str[0:4]
        df['path'] = df.displayId.str[10:13]
        df['row'] = df.displayId.str[13:16]
        df['pathrow'] = df.displayId.str[10:16]
        df['tier'] = df.displayId.str[-2:]
        df['date'] = pd.to_datetime(df.displayId.str[17:21] +df.displayId.str[21:23] + df.displayId.str[23:25])
        df['year'] = df.displayId.str[17:21]
        return df

    def logout(self):
        logout_endpoint = '/'.join([self.endpoint, 'logout'])
        r = requests.post(logout_endpoint,data={'jsonRequest': json.dumps({'apiKey': self.api_key})})
        if r.json()['errorCode'] is not None:
            return False
        return True

class Espa(object):

    '''
     Adapted from Jake Brinkmann
     Source : https://github.com/USGS-EROS/espa-api/blob/master/examples/api_demo.py
    '''

    def __init__(self, username, password, scenelist, out ,  overwrite=True, check_complete=True):
        self.user = username
        self.password= password
        self.host = 'https://espa.cr.usgs.gov/api/v1/'
        self.scenelist = scenelist
        self.path = out
        self.overwrite = overwrite
        self.check_complete = check_complete

    def espa_api(self, endpoint, verb='get', body=None, uauth=None):

        auth_tup = uauth if uauth else (self.user, self.password)
        response = getattr(requests, verb)(self.host + endpoint, auth=auth_tup, json=body)
        print('{} {}'.format(response.status_code, response.reason))
        data = response.json()
        if isinstance(data, dict):
            messages = data.pop("messages", None)
            if messages:
                print(json.dumps(messages, indent=4))
        try:
            response.raise_for_status()
        except Exception as e:
            print(e)
            return None
        else:
            return data

    def order(self):
        order=self.espa_api('/available-products', body=dict(inputs=self.scenelist),uauth=(self.user, self.password))
        return(order)

    def chunks(l, n):
        for i in range(0, len(l), n):
            yield l[i:i + n]

    def place_order(self, projection ={ "lonlat": None }, frmt= 'gtiff', resampling='cc', note='odc', products = ['sr','pixel_qa']):
        order=self.order()
        for sensor in order.keys():
            order[sensor]['products'] = products

        if 'date_restricted' in order:
            dr = order['date_restricted']
            if 'date_restricted' in order: del order['date_restricted']
            for prod in products:
                if prod in dr:
                    dr_list= dr[prod]
                    for sensor in order.keys():
                        order[sensor]['inputs'] = [e for e in order[sensor]['inputs'] if e not in dr_list]

        order['projection'] = { "lonlat": None }
        order['format'] = frmt
        order['resampling_method'] = resampling
        order['note'] = note

        resp = self.espa_api(endpoint='/order', verb='post', body=order,uauth=(self.user, self.password))
        self.orderid = resp['orderid']

    def set_orderid(self, orderid):
        self.orderid = orderid

    def order_status(self):
        return self.espa_api('order-status/{}'.format(self.orderid),uauth=(self.user, self.password))

    def complete(self):
        is_complete = self.order_status()
        return True if is_complete['status'] == 'complete' else False

    def items_status(self):
        item_status = self.espa_api('item-status/{}'.format(self.orderid),uauth=(self.user, self.password))
        return item_status[self.orderid]

    def urls_completed(self):
        item_list = self.items_status()
        url_list = [x['product_dload_url'] for x in item_list
                    if x['status'] == 'complete']
        return url_list

    def url_retrieve(self,url, filename, overwrite=True, check_complete=True):
        if os.path.isfile(filename) and not overwrite:
            if not check_complete:
                return filename
            r0 = requests.head(url)
            if os.path.getsize(filename) == int(r0.headers['Content-Length']): # file size matches
                return filename
        r = requests.get(url, stream=True)
        with open(filename, 'wb') as f:
            for chunk in r.iter_content(chunk_size=1024):
                if chunk:
                    f.write(chunk)
        return filename


    def download_all_complete(self, url):
        filename = url.split('/')[-1]
        print('Downloading: ' + filename)
        try:
            dst = os.path.join(self.path, filename)
            self.url_retrieve(url, dst, overwrite=self.overwrite,
                                 check_complete=self.check_complete)
        except Exception as e:
            print('%s skipped. Reason: %s' % (filename, e))


    def download(self, num_cores = 4, check_delay=60):
        print('Check order status ...')
        finished = self.complete()
        while finished == False:
            print('Wait until order is finished ...')
            time.sleep(check_delay)
            finished = self.complete()

        print('Start downloading ...')
        self.urls = self.urls_completed()
        pool = multiprocessing.Pool(num_cores)
        pool.map(self.download_all_complete,  self.urls)
        pool.close()
        pool.join()
        espa_complete = self.urls_completed()

        print('Download completed')

        return  espa_complete
