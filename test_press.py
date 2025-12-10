from press_utils.pipeline import eval_pressure
import argparse
# 使用--task参数传入任务描述
# python test_press.py --task "/home/j/桌面/agent_design/press_utils/data/vtk_in/F_S_WWS_WM_145_gt.vtk",
      
import vtk

def vtk_to_vtp(input_path: str, output_path: str, binary: bool = True) -> None:
    """
    将 VTK Legacy (.vtk) 数据转为 VTP (.vtp, XML PolyData)。
    若输入不是 PolyData，将通过 vtkGeometryFilter 提取表面生成 PolyData。

    :param input_path: 输入 .vtk 文件路径
    :param output_path: 输出 .vtp 文件路径
    :param binary: 是否以二进制写出（默认 True，体积更小）
    """
    # 读取任意 VTK Legacy DataSet
    reader = vtk.vtkDataSetReader()
    reader.SetFileName(input_path)
    reader.Update()
    dataset = reader.GetOutput()
    if dataset is None or dataset.GetNumberOfPoints() == 0 and dataset.GetNumberOfCells() == 0:
        raise ValueError(f"读取失败或数据为空: {input_path}")

    # 确保为 PolyData
    if isinstance(dataset, vtk.vtkPolyData):
        poly = dataset
    else:
        geom = vtk.vtkGeometryFilter()
        geom.SetInputData(dataset)
        geom.Update()
        poly = geom.GetOutput()
        if poly is None or poly.GetNumberOfPoints() == 0:
            raise ValueError("无法从输入数据中提取 PolyData 表面")

    # 写出为 .vtp
    writer = vtk.vtkXMLPolyDataWriter()
    writer.SetFileName(output_path)
    writer.SetInputData(poly)
    writer.SetDataModeToBinary() if binary else writer.SetDataModeToAscii()
    if writer.Write() == 0:
        raise IOError(f"写出失败: {output_path}")

def main(args):
    

    path = eval_pressure(input_vtk=args.task, output_vtk=args.output_dir, model_path=args.model_path)
    output_vtp = path.replace('.vtk', '.vtp')
    vtk_to_vtp(path, output_vtp, binary=True)
    print(f"{output_vtp}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run pressure field prediction using press_utils.pipeline.eval_pressure")
    parser.add_argument(
        "--task",
        type=str,
        default="/home/j/桌面/agent_design/press_utils/data/vtk_in/output.vtk",
        help="Path to the input VTK file",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="/home/j/桌面/agent_design/press_utils/data/output",
        help="Directory to save prediction outputs",
    )
    parser.add_argument(
        "--model_path",
        type=str,
        default="/home/j/桌面/agent_design/press_utils/model_300.pth",
        help="Path to the trained model checkpoint",
    )
    args = parser.parse_args()
    main(args)