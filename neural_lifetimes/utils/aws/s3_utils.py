import logging
from typing import Tuple

import boto3
from botocore.exceptions import ClientError


def decompose_s3_url(save_loc: str) -> Tuple[str, str]:
    assert save_loc[:5] == "s3://"
    save_loc = save_loc[5:]
    bucket = save_loc.split("/")[0]
    s3_file_name = save_loc[(len(bucket) + 1) :]
    return bucket, s3_file_name


def file_exists_in_s3(s3_url):
    bucket, s3_file_name = decompose_s3_url(s3_url)
    s3_resource = boto3.resource("s3")

    bucket_con = s3_resource.Bucket(bucket)
    obj = list(bucket_con.objects.filter(Prefix=s3_file_name))
    return len(obj) > 0


def save_file_to_s3(local_file_name: str, s3_url: str):
    bucket, s3_file_name = decompose_s3_url(s3_url)
    s3_client = boto3.client("s3")
    try:
        return s3_client.upload_file(local_file_name, bucket, s3_file_name)
    except ClientError as e:
        logging.error(e)
        raise


def get_file_from_s3(s3_url: str, local_file_name: str):
    bucket, s3_file_name = decompose_s3_url(s3_url)
    s3 = boto3.resource("s3")
    try:
        return s3.Bucket(bucket).download_file(s3_file_name, local_file_name)
    except ClientError as e:
        if e.response["Error"]["Code"] == "404":
            print("The object does not exist.")
        else:
            raise
