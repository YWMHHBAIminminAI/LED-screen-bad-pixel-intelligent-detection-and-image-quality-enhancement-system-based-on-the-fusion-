# DepartmentTestBaseOnGoland
设备监控压测系统


// 背压控制核心
func ProcessStream(input <-chan DeviceData) {
    sem := make(chan struct{}, 10000) // 控制1万并发
    for data := range input {
        sem <- struct{}{}
        go func(d DeviceData) {
            processDevice(d)
            <-sem
        }(data)
    }
}


# 启动10万设备模拟
go run simulator.go -devices=100000
