// Linked into every assembly that is compiled against the publicized reference (host\publicized\sts2.dll).
// The CoreCLR honours this attribute: the assembly may touch non-public members of sts2.dll directly
// (a plain ldfld / call, no reflection). The game file on disk is never modified.
using System;

[assembly: System.Runtime.CompilerServices.IgnoresAccessChecksTo("sts2")]

namespace System.Runtime.CompilerServices
{
    [AttributeUsage(AttributeTargets.Assembly, AllowMultiple = true)]
    internal sealed class IgnoresAccessChecksToAttribute : Attribute
    {
        public IgnoresAccessChecksToAttribute(string assemblyName)
        {
            AssemblyName = assemblyName;
        }

        public string AssemblyName { get; }
    }
}
