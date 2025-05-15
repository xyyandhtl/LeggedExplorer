from isaacsim import SimulationApp

simulation_app = SimulationApp()

from pxr import Usd, UsdShade, Sdf, UsdGeom

robot = 'a1'
usd_path = f'/home/lenovo/Documents/usd/robot/{robot}/'

# 获取当前的 USD Stage
stage = Usd.Stage.Open(f"{usd_path}/{robot}.usd")  # 替换为您的 USD 文件路径

# 获取目标网格
mesh = UsdGeom.Mesh(stage.GetPrimAtPath(f"/{robot}/trunk/visuals"))  # 替换为您的网格路径

# 创建材质路径
material_path = Sdf.Path(f"/{robot}/CetcMaterial")
material = UsdShade.Material.Define(stage, material_path)

# 创建 PBR Shader
shader = UsdShade.Shader.Define(stage, material_path.AppendChild("PBRShader"))
shader.CreateIdAttr("UsdPreviewSurface")

# 创建纹理节点
texture_shader = UsdShade.Shader.Define(stage, material_path.AppendChild("DiffuseTexture"))
texture_shader.CreateIdAttr("UsdUVTexture")
texture_shader.CreateInput("file", Sdf.ValueTypeNames.Asset).Set(f"{usd_path}/textures/4.png")  # 替换为您的纹理图片路径
texture_shader.CreateInput("wrapS", Sdf.ValueTypeNames.Token).Set("repeat")
texture_shader.CreateInput("wrapT", Sdf.ValueTypeNames.Token).Set("repeat")

# 创建 UV Reader
uv_reader = UsdShade.Shader.Define(stage, material_path.AppendChild("PrimvarReader"))
uv_reader.CreateIdAttr("UsdPrimvarReader_float2")
uv_reader.CreateInput("varname", Sdf.ValueTypeNames.Token).Set("st")
uv_output = uv_reader.CreateOutput("result", Sdf.ValueTypeNames.Float2)

# 连接 UV Reader 到纹理节点
texture_shader.CreateInput("st", Sdf.ValueTypeNames.Float2).ConnectToSource(uv_output)

# 获取纹理的输出口
texture_rgb_output = texture_shader.CreateOutput("rgb", Sdf.ValueTypeNames.Float3)

# 然后连接
shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).ConnectToSource(texture_rgb_output)

# Shader 的 surface 输出
shader_surface_output = shader.CreateOutput("surface", Sdf.ValueTypeNames.Token)

# 连接材质输出
material.CreateSurfaceOutput().ConnectToSource(shader_surface_output)


# 将材质绑定到网格
UsdShade.MaterialBindingAPI(mesh).Bind(material)

# 保存修改后的 USD 文件（可以覆盖原文件或另存为新文件）
stage.GetRootLayer().Export(f"{usd_path}/{robot}_.usd")
