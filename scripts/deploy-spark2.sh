#!/usr/bin/env bash

#
# Copyright 2016 The BigDL Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

#
# For internal use to deploy bigdl maven artifacts from jenkins server to sonatype
#

set -e

# Check java
if type -p java>/dev/null; then
    _java=java
else
    echo "Java is not installed"
    exit 1
fi

# We deploy artifacts build from jdk 1.8
JDK_VERSION=$("$_java" -version 2>&1 | awk -F '"' '/version/ {print $2}')
if [[ "$JDK_VERSION" > "1.9" ]]; then
    echo Require a jdk 1.8 version
    exit 1
fi
if [[ "$JDK_VERSION" < "1.8" ]]; then
    echo Require a jdk 1.8 version
    exit 1
fi
export MAVEN_OPTS="-Xmx2g -XX:ReservedCodeCacheSize=512m"

# Check if mvn installed
MVN_INSTALL=$(which mvn 2>/dev/null | grep mvn | wc -l)
if [ $MVN_INSTALL -eq 0 ]; then
  echo "MVN is not installed. Exit"
  exit 1
fi

cd spark
# Append Spark platform variable to bigdl artifact name and spark-version artifact name
mv dl/pom.xml dl/pom.xml.origin
cat dl/pom.xml.origin | sed 's/<artifactId>bigdl<\/artifactId>/<artifactId>bigdl-${SPARK_PLATFORM}<\/artifactId>/' | \
    sed 's/<artifactId>${spark-version.project}<\/artifactId>/<artifactId>${spark-version.project}-${SPARK_PLATFORM}<\/artifactId>/' > dl/pom.xml
mv spark-version/1.5-plus/pom.xml spark-version/1.5-plus/pom.xml.origin
cat spark-version/1.5-plus/pom.xml.origin | sed 's/<artifactId>1.5-plus<\/artifactId>/<artifactId>1.5-plus-${SPARK_PLATFORM}<\/artifactId>/' > spark-version/1.5-plus/pom.xml
mv spark-version/2.0/pom.xml spark-version/2.0/pom.xml.origin
cat spark-version/2.0/pom.xml.origin | sed 's/<artifactId>2.0<\/artifactId>/<artifactId>2.0-${SPARK_PLATFORM}<\/artifactId>/' > spark-version/2.0/pom.xml

function deploy {
    mvn clean install -DskipTests  -Dspark.version=$1 -DSPARK_PLATFORM=$2 $3
    cd spark-version && mvn install -DskipTests  -Dspark.version=$1 -DSPARK_PLATFORM=$2 $3 && cd ..
    cd dl && mvn install -DskipTests  -Dspark.version=$1 -DSPARK_PLATFORM=$2 $3 && cd ..
    cd dist && mvn install -DskipTests -Dspark.version=$1 -DSPARK_PLATFORM=$2 $3 && cd ..
}

deploy 2.1.1 SPARK_2.1 '-P spark_2.x -Dscala.version=2.10.6 -Dscala.major.version=2.10'


mv dl/pom.xml.origin dl/pom.xml
mv spark-version/1.5-plus/pom.xml.origin spark-version/1.5-plus/pom.xml
mv spark-version/2.0/pom.xml.origin spark-version/2.0/pom.xml
