# Project Dependencies

## System Requirements

### Java
- JDK 17
```bash
sudo apt install openjdk-17-jdk
```

### Python
- Python 3.11.8 (Core functionality)
- Python 3.12 (Streamlit reporting)
```bash
sudo apt install python3.11
sudo apt install python3.12
```

### Big Data Framework
- Apache Hadoop 3.4.0
```bash
wget https://downloads.apache.org/hadoop/common/hadoop-3.4.0/hadoop-3.4.0.tar.gz
tar -xzf hadoop-3.4.0.tar.gz
```

- Apache Spark 3.5.3
```bash
wget https://downloads.apache.org/spark/spark-3.5.3/spark-3.5.3-bin-hadoop3.tgz
tar -xzf spark-3.5.3-bin-hadoop3.tgz
```

### Additional Setup
- For Streamlit reporting:
```bash
python3.12 -m venv streamlit_env
source streamlit_env/bin/activate
pip install streamlit
```
