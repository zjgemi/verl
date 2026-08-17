#!/usr/bin/env python3
"""
LBG Sandbox Manager - 自动化沙盒操作脚本

功能：
1. 创建模板（指定镜像和机型）
2. 创建沙盒
3. 上传本地文件到沙盒
4. 执行命令
5. 下载文件到本地
6. 构建镜像
7. 清理沙盒（无论成功失败）
"""

import subprocess
import sys
import os
import json
from typing import List, Optional, Dict, Any
from pathlib import Path


class LBGSandboxError(Exception):
    """LBG沙盒操作异常"""
    pass


class LBGSandboxManager:
    def __init__(self, verbose: bool = False):
        self.template_id: Optional[str] = None
        self.sandbox_id: Optional[str] = None
        self.verbose = verbose

    def _log(self, msg: str):
        if self.verbose:
            print(msg)

    def _run_command(self, cmd: List[str], error_msg: str = None, parse_json: bool = True) -> Dict[str, Any]:
        """执行命令并返回解析后的JSON输出"""
        # 添加 --json 选项
        if parse_json and "--json" not in cmd:
            cmd = cmd + ["--json"]

        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                check=False  # 不自动检查返回码，手动处理
            )

            if parse_json:
                try:
                    # 处理可能包含升级提示的输出
                    output = result.stdout.strip()

                    # 如果输出包含多个JSON对象（用换行分隔），分别解析
                    lines = output.split('\n')
                    json_objects = []
                    current_json = ""
                    brace_count = 0

                    for line in lines:
                        for char in line:
                            if char == '{':
                                brace_count += 1
                            elif char == '}':
                                brace_count -= 1
                            current_json += char

                        if brace_count == 0 and current_json.strip():
                            try:
                                obj = json.loads(current_json.strip())
                                json_objects.append(obj)
                                current_json = ""
                            except json.JSONDecodeError:
                                current_json += '\n'
                        else:
                            current_json += '\n'

                    # 过滤掉升级提示，返回第一个有效的响应
                    for obj in json_objects:
                        if obj.get("kind") != "upgrade_available":
                            # 检查是否有错误
                            if (obj.get("error") or result.returncode != 0) and error_msg:
                                error_detail = obj.get("error", "未知错误")
                                raise LBGSandboxError(
                                    f"{error_msg}\n命令: {' '.join(cmd)}\n"
                                    f"返回码: {result.returncode}\n"
                                    f"错误: {error_detail}"
                                )
                            return obj

                    if json_objects:
                        return json_objects[-1]

                    raise LBGSandboxError(f"未找到有效的JSON输出: {output}")

                except json.JSONDecodeError as e:
                    raise LBGSandboxError(
                        f"解析JSON输出失败: {e}\n"
                        f"原始输出: {result.stdout}"
                    )
            else:
                if result.returncode != 0 and error_msg:
                    raise LBGSandboxError(
                        f"{error_msg}\n命令: {' '.join(cmd)}\n"
                        f"返回码: {result.returncode}\n"
                        f"标准输出: {result.stdout}"
                        f"标准错误: {result.stderr}"
                    )
                return {"output": result.stdout.strip()}

        except FileNotFoundError:
            raise LBGSandboxError("未找到lbg命令，请确保已安装lbg sdbx工具")

    def create_template(self, image: str, sku_name: str, name: str) -> str:
        """创建模板

        Args:
            image: 镜像名称，如 'registry.dp.tech/dptech/abacus:LTSv3.10.1'
            sku_name: SKU名称，如 'c4_m8_cpu' 或 'c16_m64_1 * NVIDIA 5090'
            name: 模板名称（必需）
        """
        self._log(f"创建模板: 镜像={image}, SKU={sku_name}, 名称={name}")
        cmd = ["lbg", "sdbx", "template", "create", "--image", image, "--sku-name", sku_name, "--name", name]

        try:
            result = self._run_command(cmd, "创建模板失败")
        except LBGSandboxError as e:
            # 如果模板已存在，直接使用该模板
            if "already exists" in str(e):
                self._log(f"⚠ 模板已存在，将使用现有模板: {name}")
                self.template_id = name
                return self.template_id
            raise

        # 根据文档，模板使用 name 字段作为标识
        self.template_id = result.get("name") or name
        if not self.template_id:
            raise LBGSandboxError(f"无法从输出中提取模板名称: {result}")

        self._log(f"✓ 模板创建成功: {self.template_id}")
        return self.template_id

    def delete_template(self, name: str):
        """删除模板

        Args:
            name: 模板名称（必需）
        """
        print(f"删除模板: 名称={name}")
        cmd = ["lbg", "sdbx", "template", "rm", "--force", name]
        result = self._run_command(cmd, "删除模板失败", parse_json=False)
        print(f"✓ 模板删除成功: {name}")

    def create_sandbox(self, template_name: Optional[str] = None, project_id: Optional[str] = None, never_timeout: bool = False, timeout: Optional[int] = None) -> str:
        """创建沙盒

        Args:
            template_name: 模板名称（不是数字ID），如果不指定则使用之前创建的模板
            project_id: 可选的项目ID，用于计费到项目预算而非个人钱包
            never_timeout: 是否禁用自动销毁（需要手动关闭沙盒）
            timeout: 沙盒自动销毁时间（秒），与 never_timeout 互斥
        """
        template = template_name or self.template_id
        if not template:
            raise LBGSandboxError("未指定模板名称")

        self._log(f"创建沙盒: 模板={template}")
        cmd = ["lbg", "sdbx", "create", template]
        if project_id:
            cmd.extend(["--project-id", project_id])
        if never_timeout:
            cmd.append("--never-timeout")
        elif timeout is not None:
            cmd.extend(["--timeout", str(timeout)])

        result = self._run_command(cmd, "创建沙盒失败")

        # 根据文档，返回 sandboxID 字段
        self.sandbox_id = result.get("sandboxID") or result.get("id")
        if not self.sandbox_id:
            raise LBGSandboxError(f"无法从输出中提取沙盒ID: {result}")

        self._log(f"✓ 沙盒创建成功: {self.sandbox_id}")
        return self.sandbox_id

    def upload_files(self, local_files: List[str], remote_dir: str = "/tmp/workspace") -> None:
        """上传文件到沙盒

        Args:
            local_files: 本地文件或目录列表
            remote_dir: 远程目标目录
        """
        if not self.sandbox_id:
            raise LBGSandboxError("沙盒未创建")

        for local_file in local_files:
            if not os.path.exists(local_file):
                raise LBGSandboxError(f"本地文件不存在: {local_file}")

            # 获取文件名
            filename = os.path.basename(local_file)
            # 构建完整的远程路径
            remote_path = f"{remote_dir}/{filename}"

            print(f"上传文件: {local_file} -> {remote_path}")
            # 根据文档使用 files write --source
            cmd = ["lbg", "sdbx", "files", "write", "--source", local_file, self.sandbox_id, remote_path]
            self._run_command(cmd, f"上传文件失败: {local_file}")
            print(f"✓ 文件上传成功: {local_file}")

    def execute_command(self, command: str, background: bool = False, timeout: Optional[int] = None, user: Optional[str] = None) -> str:
        """在沙盒中执行命令

        Args:
            command: 要执行的命令
            background: 是否后台执行
            timeout: 超时时间（秒），0表示无限制，None表示使用默认值
        """
        if not self.sandbox_id:
            raise LBGSandboxError("沙盒未创建")

        # print(f"执行命令: {command}")
        # 根据文档，exec 使用位置参数传递命令
        cmd = ["lbg", "sdbx", "exec"]
        if background:
            cmd.append("--background")
        if timeout is not None:
            cmd.extend(["--timeout", str(timeout)])
        if user is not None:
            cmd.extend(["--user", user])
        cmd.append(self.sandbox_id)
        cmd.append(command)

        result = self._run_command(cmd, f"执行命令失败: {command}", parse_json=False)

        output = result.get("output", "")
        self._log(f"✓ 命令执行成功")
        if output:
            self._log(f"输出:\n{output}")
        return output

    def read_file(self, remote_file: str) -> None:
        """读取文件内容

        Args:
            remote_file: 远程文件路径
        """
        if not self.sandbox_id:
            raise LBGSandboxError("沙盒未创建")

        cmd = ["lbg", "sdbx", "files", "read", self.sandbox_id, remote_file]
        result = self._run_command(cmd, f"读取文件失败: {remote_file}", parse_json=False)
        return result["output"]

    def download_file(self, remote_file: str, local_path: str, binary: bool = False) -> None:
        """从沙盒下载文件

        Args:
            remote_file: 远程文件路径
            local_path: 本地保存路径
            binary: 是否以二进制模式下载（用于tar/zip等二进制文件，默认False使用text模式）
        """
        if not self.sandbox_id:
            raise LBGSandboxError("沙盒未创建")

        print(f"下载文件: {remote_file} -> {local_path}" + (" (binary)" if binary else ""))

        # 确保本地目录存在
        local_dir = os.path.dirname(local_path)
        if local_dir:
            os.makedirs(local_dir, exist_ok=True)

        # 根据文档使用 files read，二进制文件必须使用 --format bytes
        cmd = ["lbg", "sdbx", "files", "read"]
        if binary:
            cmd.extend(["--format", "bytes"])
        cmd.extend([self.sandbox_id, remote_file, "--output", local_path])
        self._run_command(cmd, f"下载文件失败: {remote_file}", parse_json=False)
        print(f"✓ 文件下载成功: {local_path}")

    def commit_image(self, image_name: str, project_id: str, description: str = "") -> str:
        """基于沙盒提交镜像（异步操作）

        Args:
            image_name: 镜像名称
            project_id: 项目ID（必需，用于镜像存储计费）
            description: 镜像描述

        Returns:
            commit_id: 提交任务ID，可用于轮询状态
        """
        if not self.sandbox_id:
            raise LBGSandboxError("沙盒未创建")

        print(f"提交镜像: {image_name}")
        cmd = [
            "lbg", "sdbx", "image", "commit",
            "--sandbox-id", self.sandbox_id,
            "--name", image_name,
            "--project-id", project_id
        ]
        if description:
            cmd.extend(["--desc", description])

        result = self._run_command(cmd, "提交镜像失败")

        # 尝试从不同位置提取 ID
        commit_id = result.get("id")
        if not commit_id and "response" in result:
            commit_id = result["response"].get("id")
        if not commit_id:
            raise LBGSandboxError(f"无法从输出中提取提交ID: {result}")

        print(f"✓ 镜像提交任务已创建: {commit_id}")
        return str(commit_id)

    def wait_for_image_commit(self, commit_id: str, poll_interval: int = 10, max_wait: int = 600) -> str:
        """等待镜像提交完成

        Args:
            commit_id: 提交任务ID
            poll_interval: 轮询间隔（秒）
            max_wait: 最大等待时间（秒）

        Returns:
            image_url: 镜像URL
        """
        import time

        print(f"等待镜像提交完成: {commit_id}")
        start_time = time.time()

        while time.time() - start_time < max_wait:
            cmd = ["lbg", "sdbx", "image", "get", commit_id]
            result = self._run_command(cmd, "查询镜像状态失败")

            status = result.get("status")
            # status: 0=creating 1=pending 2=success 3=failed
            if status == 2:
                image_url = result.get("imageUrl", "")
                print(f"✓ 镜像提交成功: {image_url}")
                return image_url
            elif status == 3:
                error_msg = result.get("errorMsg", "未知错误")
                raise LBGSandboxError(f"镜像提交失败: {error_msg}")

            print(f"  镜像提交中 (状态: {status})...")
            time.sleep(poll_interval)

        raise LBGSandboxError(f"镜像提交超时（等待了 {max_wait} 秒）")

    def list_processes(self) -> list:
        """列出沙盒中正在运行的进程"""
        if not self.sandbox_id:
            raise LBGSandboxError("沙盒未创建")

        cmd = ["lbg", "sdbx", "ps", self.sandbox_id, "--json"]
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, check=False)
            if result.returncode != 0:
                raise LBGSandboxError(f"查询进程列表失败: {result.stderr}")

            return json.loads(result.stdout.strip())
        except json.JSONDecodeError as e:
            raise LBGSandboxError(f"解析进程列表失败：{e}")

    def wait_for_processes(self, max_wait: int = 3600, poll_interval: int = 10,
                           label: str = "任务") -> None:
        """等待沙盒中所有进程结束

        Args:
            max_wait: 最大等待时间（秒）
            poll_interval: 轮询间隔（秒）
            label: 状态消息中的任务名称
        """
        import time

        print(f"  等待 {label} 完成...", file=sys.stderr)
        elapsed = 0

        while elapsed < max_wait:
            if elapsed < poll_interval:
                time.sleep(1)
                elapsed += 1
            else:
                time.sleep(poll_interval)
                elapsed += poll_interval

            try:
                processes = self.list_processes()
                if not processes:
                    print(f"  {label} 完成 ({elapsed}秒)", file=sys.stderr)
                    return
            except Exception as e:
                print(f"  检查 {label} 状态出错: {e}", file=sys.stderr)

        raise LBGSandboxError(f"{label} 超时（{max_wait}秒）")

    def close_sandbox(self) -> None:
        """关闭沙盒"""
        if not self.sandbox_id:
            print("无需关闭沙盒（未创建）")
            return

        self._log(f"关闭沙盒: {self.sandbox_id}")
        try:
            # 根据文档使用 kill 命令
            cmd = ["lbg", "sdbx", "kill", self.sandbox_id]
            self._run_command(cmd, "关闭沙盒失败")
            self._log(f"✓ 沙盒已关闭: {self.sandbox_id}")
        except LBGSandboxError as e:
            print(f"⚠ 关闭沙盒时出错: {e}", file=sys.stderr)

    def run_workflow(
        self,
        image: str,
        sku_name: str,
        template_name: str,
        upload_files: List[str],
        commands: List[str],
        download_files: List[tuple[str, str]],
        commit_image_name: Optional[str] = None,
        commit_image_project_id: Optional[str] = None,
        commit_image_description: str = "",
        project_id: Optional[str] = None
    ) -> None:
        """执行完整工作流

        Args:
            image: 镜像名称
            sku_name: SKU名称
            template_name: 模板名称（必需）
            upload_files: 要上传的本地文件列表
            commands: 要执行的命令列表
            download_files: 要下载的文件列表 [(远程路径, 本地路径), ...]
            commit_image_name: 可选，提交镜像的名称
            commit_image_project_id: 提交镜像时的项目ID（如果提交镜像则必需）
            commit_image_description: 镜像描述
            project_id: 可选，沙盒计费项目ID
        """
        try:
            # 1. 创建模板
            self.create_template(image, sku_name, template_name)

            # 2. 创建沙盒
            self.create_sandbox(project_id=project_id)

            # 3. 上传文件
            if upload_files:
                self.upload_files(upload_files)

            # 4. 执行命令
            for cmd in commands:
                self.execute_command(cmd)

            # 5. 下载文件
            for remote_file, local_path in download_files:
                self.download_file(remote_file, local_path)

            # 6. 提交镜像
            if commit_image_name:
                if not commit_image_project_id:
                    raise LBGSandboxError("提交镜像需要指定 commit_image_project_id")
                commit_id = self.commit_image(
                    commit_image_name,
                    commit_image_project_id,
                    commit_image_description
                )
                self.wait_for_image_commit(commit_id)

            print("\n✓ 所有操作完成")

        except Exception as e:
            print(f"\n✗ 操作失败: {e}", file=sys.stderr)
            raise

        finally:
            # 7. 清理沙盒
            self.close_sandbox()


def main():
    """示例用法 - 完整的工作流演示"""
    manager = LBGSandboxManager()

    # 完整配置示例
    config = {
        "image": "registry.dp.tech/dptech/ubuntu:22.04-py3.10",
        "sku_name": "c2_m4_cpu",  # 或使用 GPU: "c16_m64_1 * NVIDIA 5090"
        "template_name": "my-template",  # 必需
        "upload_files": [
            "./test_script.py",   # 要上传的脚本
            "./test_data.txt"     # 要上传的数据
        ],
        "commands": [
            "ls -la /tmp/workspace",
            "cat /tmp/workspace/test_data.txt",
            "python3 /tmp/workspace/test_script.py",
            "cat /tmp/workspace/output.txt"
        ],
        "download_files": [
            ("/tmp/workspace/output.txt", "./downloaded_output.txt")
        ],
        "commit_image_name": "my-custom-image",
        "commit_image_project_id": "10229",  # 必需，替换为实际的项目ID
        "commit_image_description": "Custom image with script and data",
        "project_id": None  # 可选，沙盒计费项目ID
    }

    try:
        manager.run_workflow(**config)
        print("\n" + "="*50)
        print("工作流执行完成！")
        print("="*50)
        sys.exit(0)
    except Exception as e:
        print(f"程序异常退出: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
