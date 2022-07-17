# Module ASP

Predicting action for picking entangled wire harness

1. Create and generate `asp.proto`
```
python -m grpc_tools.protoc -I./ --python_out=./ --grpc_python_out=./ asp.proto
or
python code_gen.py
```

2. Start the server 

```
python server.py 
```