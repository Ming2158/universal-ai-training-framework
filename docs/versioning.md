# UATF Versioning and Release Rules

UATF uses Semantic Versioning in the form `MAJOR.MINOR.PATCH`. Git release tags use a `v` prefix, such as `v1.0.0`.

## Choose the version

- **PATCH** (`1.0.1`, `1.0.2`): compatible bug fixes, small improvements, and documentation corrections that do not change the main features or normal usage.
- **MINOR** (`1.1.0`, `1.10.0`): compatible features or expanded capabilities. The number can grow beyond 9; `1.9.0` can be followed by `1.10.0`.
- **MAJOR** (`2.0.0`, `3.0.0`): incompatible core interfaces, major architecture or workflow changes, or a new major development stage.

Choose by compatibility and user impact, not by the number of changed lines. After `1.0.0`, a fix normally moves to `1.0.1`, a compatible feature to `1.1.0`, and an incompatible change to `2.0.0`.

## Record a release

Before a release, update the version in both `pyproject.toml` and `uatf/__init__.py`, add dated notes to `CHANGELOG.md`, and keep the release notes consistent with verified behavior. Run the applicable tests, create a release commit, then add an annotated `vMAJOR.MINOR.PATCH` tag to that commit. Published version tags are permanent and must not be moved or reused. Do not push or publish a remote release without authorization.

Until the next release is finalized, the version fields continue to identify the latest released version; unreleased work on `main` is not a release.

The existing `v1.0.0` tag points to commit `378167c`. Changes after that tag are unreleased work. The bilingual interactive CLI wizard is an additive feature and belongs in the next MINOR release; it is recorded under `Unreleased` in the changelog.

## 版本选择

UATF 使用 `主版本号.次版本号.修订版本号`，Git 标签使用 `v` 前缀，例如 `v1.0.0`。

- **PATCH 修订版本**（`1.0.1`、`1.0.2`）：修复问题、小型优化和文档修正，不改变主要功能和常规用法。
- **MINOR 次版本**（`1.1.0`、`1.10.0`）：增加兼容的新功能或扩展现有能力。版本号可以超过 9，`1.9.0` 之后可以是 `1.10.0`。
- **MAJOR 主版本**（`2.0.0`、`3.0.0`）：核心接口不兼容、重大架构或使用流程改变，或项目进入新的重大开发阶段。

版本号根据兼容性和用户影响决定，不根据修改行数决定。`1.0.0` 之后，普通修复通常发布为 `1.0.1`，兼容新功能发布为 `1.1.0`，不兼容变更发布为 `2.0.0`。

## 发布记录

发布前同步更新 `pyproject.toml` 和 `uatf/__init__.py` 中的版本号，在 `CHANGELOG.md` 记录发布日期和已验证的内容，并确保发布说明与实际行为一致。运行适当的测试后，创建发布提交，再为该提交添加带说明的 `v主版本号.次版本号.修订版本号` 标签。已发布的版本标签不可移动或重复使用。未经授权不得推送或发布远程 Release。

下一个正式版本定稿前，版本字段继续表示最近一次已发布的版本；`main` 上尚未发布的改动不代表一次正式发布。

现有 `v1.0.0` 标签指向提交 `378167c`。标签之后的改动属于尚未发布的内容；双语交互式 CLI 向导属于兼容新功能，应进入下一个 MINOR 版本，并记录在更新日志的 `Unreleased` 部分。
